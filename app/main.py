from __future__ import annotations

import datetime as dt
import signal
import sys
from typing import Any

from app.config import Settings
from app.discord_notifier import DiscordNotifier
from app.execution_engine import build_limit_order_from_signal
from app.kalshi_client import KalshiClient, KalshiAPIError
from app.logger import get_logger
from app.market_data import extract_markets, summarize_market
from app.performance import summarize_decisions
from app.portfolio_tracker import PortfolioSnapshot, parse_balance_payload
from app.risk_engine import evaluate_risk
from app.scheduler import run_loop
from app.signal_engine import evaluate_signal
from app.social_signal import evaluate_social_signal
from app.state_store import StateStore


class BotApp:
    def __init__(self) -> None:
        self.settings = Settings()
        self.settings.validate()
        self.logger = get_logger("kalshi-bot", self.settings.LOG_LEVEL)
        self.notifier = DiscordNotifier(self.settings.DISCORD_WEBHOOK_URL, logger=self.logger)
        self.store = StateStore(self.settings.DB_PATH)
        self.client = KalshiClient(
            api_key_id=self.settings.KALSHI_API_KEY_ID,
            private_key_pem=self.settings.KALSHI_PRIVATE_KEY_PEM,
            base_url=self.settings.base_url,
            dry_run=self.settings.DRY_RUN,
            logger=self.logger,
        )
        self._stopping = False

    def shutdown(self) -> None:
        self._stopping = True
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self.store.close()
        except Exception:
            pass

    def startup_message(self) -> None:
        msg = (
            f"🚀 Kalshi bot started | env={self.settings.KALSHI_ENV} "
            f"| dry_run={self.settings.DRY_RUN} | interval={self.settings.SCAN_INTERVAL_SECONDS}s"
        )
        self.logger.info(msg)
        self.notifier.send(msg)

    def _fetch_portfolio_snapshot(self) -> PortfolioSnapshot:
        snap = PortfolioSnapshot()
        try:
            bal = self.client.get_portfolio_balance()
            snap.cash_balance_dollars = parse_balance_payload(bal)
        except Exception as e:
            self.logger.warning(f"Balance fetch failed: {e}")

        # MVP: positions endpoint parsing can be tightened after seeing your exact payload
        try:
            pos = self.client.get_positions()
            if isinstance(pos, dict):
                positions = pos.get("positions") if isinstance(pos.get("positions"), list) else []
                snap.open_positions_count = len(positions)
                # rough placeholder; improve after payload inspection
                snap.open_exposure_dollars = min(
                    self.settings.MAX_OPEN_EXPOSURE_DOLLARS,
                    len(positions) * self.settings.MIN_TRADE_DOLLARS,
                )
        except Exception as e:
            self.logger.warning(f"Positions fetch failed: {e}")

        # MVP daily pnl placeholder = 0 until fills/pnl endpoint normalized
        snap.daily_pnl_dollars = 0.0
        return snap

    def _choose_market(self, markets: list[dict]) -> Any:
    candidates = []
    for m in markets[:200]:
        s = summarize_market(m)

        if s["status"].lower() not in {"active", "open", "trading", "unknown"}:
            continue

        # basic quality filters
        if s["spread_cents"] <= 0 or s["spread_cents"] > self.settings.MAX_SPREAD_CENTS:
            continue
        if s["volume"] < self.settings.MIN_RECENT_VOLUME:
            continue

        # avoid ultra-extreme dead zones for MVP strategy
        mid = s["mid_yes_cents"]
        if mid <= 2 or mid >= 98:
            continue

        candidates.append(s)

    # prioritize good spread + healthy volume + non-extreme pricing
    candidates.sort(key=lambda x: (x["spread_cents"], abs(x["mid_yes_cents"] - 50), -x["volume"]))
    return candidates[0] if candidates else None
    
    def tick(self) -> None:
        now = dt.datetime.now(dt.UTC).isoformat()
        raw = self.client.get_markets(limit=100)
        markets = extract_markets(raw)
        self.logger.info(f"Fetched markets: {len(markets)}")
        if not markets:
            return

        chosen = self._choose_market(markets)
        if not chosen:
            self.logger.info("No candidate market found.")
            return

        self.logger.info(
            f"Candidate {chosen['ticker']} | mid={chosen['mid_yes_cents']:.1f} "
            f"| spread={chosen['spread_cents']} | vol={chosen['volume']}"
        )

        sig = evaluate_signal(
            chosen,
            min_conviction=self.settings.MIN_CONVICTION_SCORE,
            max_spread_cents=self.settings.MAX_SPREAD_CENTS,
            min_volume=self.settings.MIN_RECENT_VOLUME,
        )

        social = evaluate_social_signal(
            chosen,
            enabled=self.settings.SOCIAL_SIGNAL_ENABLED,
            max_bonus=self.settings.SOCIAL_MAX_BONUS_SCORE,
        )
        final_conviction = min(100.0, sig.conviction_score + social.bonus_score)

        snapshot = self._fetch_portfolio_snapshot()

        # modest risk-on size boost only when both signal and social are supportive
        social_size_boost = 0.0
        if sig.direction != "SKIP" and social.bonus_score > 0:
            social_size_boost = min(self.settings.SOCIAL_SIZE_BOOST_MAX_PCT, social.bonus_score / 100.0)

        risk = evaluate_risk(
            conviction=final_conviction,
            current_open_exposure=snapshot.open_exposure_dollars,
            open_positions_count=snapshot.open_positions_count,
            daily_pnl=snapshot.daily_pnl_dollars,
            min_dollars=self.settings.MIN_TRADE_DOLLARS,
            max_dollars=self.settings.MAX_TRADE_DOLLARS,
            daily_max_loss_dollars=self.settings.DAILY_MAX_LOSS_DOLLARS,
            max_open_exposure_dollars=self.settings.MAX_OPEN_EXPOSURE_DOLLARS,
            max_simultaneous_positions=self.settings.MAX_SIMULTANEOUS_POSITIONS,
            social_size_boost_pct=social_size_boost,
        )

        action = "SKIP"
        order_resp = None
        reasons = sig.reasons + [f"social:{','.join(social.reasons)}", f"risk:{risk.reason}"]

        if sig.direction != "SKIP" and risk.allowed:
            order = build_limit_order_from_signal(chosen["ticker"], sig.direction, chosen, risk.stake_dollars)
            try:
                order_resp = self.client.place_order(
                    ticker=order["ticker"],
                    side=order["side"],
                    yes_no=order["yes_no"],
                    count=order["count"],
                    price_cents=order["price_cents"],
                    client_order_id=order["client_order_id"],
                )
                action = f"TRADE_{sig.direction}"
                self.logger.info(
                    f"{action} {chosen['ticker']} stake=${risk.stake_dollars:.2f} "
                    f"price={order['price_cents']}c count={order['count']} dry_run={self.settings.DRY_RUN}"
                )
                self.notifier.send(
                    f"📈 {action} `{chosen['ticker']}` | stake=${risk.stake_dollars:.2f} "
                    f"| conv={final_conviction:.1f} (base {sig.conviction_score:.1f} + social {social.bonus_score:.1f}) "
                    f"| dry_run={self.settings.DRY_RUN}"
                )
            except KalshiAPIError as e:
                action = "ERROR"
                self.logger.exception(f"Order error: {e}")
                self.notifier.send(f"⚠️ Kalshi order error: {e}")
        else:
            self.logger.info(
                f"SKIP {chosen['ticker']} | dir={sig.direction} | conv={final_conviction:.1f} "
                f"| risk={risk.reason}"
            )

        self.store.insert_decision({
            "ts": now,
            "ticker": chosen["ticker"],
            "direction": sig.direction,
            "base_conviction": sig.conviction_score,
            "social_bonus": social.bonus_score,
            "final_conviction": final_conviction,
            "stake_dollars": risk.stake_dollars if risk.allowed else 0.0,
            "action": action,
            "reasons": ";".join(reasons),
            "dry_run": self.settings.DRY_RUN,
        })

        # simple periodic summary every tick (light)
        recent = self.store.recent_decisions(20)
        summary = summarize_decisions(recent)
        self.logger.info(f"Recent summary: {summary}")

        # withdrawal alert (notification-only)
        if snapshot.cash_balance_dollars >= self.settings.WITHDRAWAL_ALERT_THRESHOLD_DOLLARS:
            self.notifier.send(
                f"💸 Cash balance appears >= ${self.settings.WITHDRAWAL_ALERT_THRESHOLD_DOLLARS:.0f}. "
                f"Consider manual withdrawal via Kalshi UI."
            )

def main() -> None:
    app = BotApp()

    def _handle_stop(signum, frame):
        app.logger.info(f"Received signal {signum}; shutting down.")
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    app.startup_message()
    run_loop(app.settings.SCAN_INTERVAL_SECONDS, app.tick, logger=app.logger)


if __name__ == "__main__":
    main()
