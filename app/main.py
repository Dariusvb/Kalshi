from __future__ import annotations

"""
Kalshi Expert Trading Bot — Aggressive EV-Maximizing Edition
============================================================
Philosophy: Find the highest-edge play each tick. Size it proportionally
to your edge using fractional Kelly (18-45%). Cut losers decisively.
Let winners breathe. Never spam low-edge trades. Never leave money on the
table when conviction is high. This bot runs like its life depends on profits.

Architecture: fully compatible with the existing app.* module set.
"""

import datetime as dt
import signal
import sys
from typing import Any, Optional

from app.config import Settings
from app.discord_notifier import DiscordNotifier
from app.execution_engine import build_limit_order_from_signal  # kept for compat
from app.kalshi_client import KalshiClient, KalshiAPIError
from app.logger import get_logger
from app.market_data import extract_markets, summarize_market
from app.paper_resolver import resolve_paper_trades
from app.performance import summarize_decisions
from app.portfolio_tracker import PortfolioSnapshot, parse_balance_payload
from app.risk_engine import evaluate_risk
from app.scheduler import run_loop
from app.signal_engine import evaluate_signal
from app.social_signal import evaluate_social_signal
from app.state_store import StateStore

try:
    from app.learning import compute_learning_adjustment
except Exception:
    compute_learning_adjustment = None  # type: ignore

try:
    from app.news_signal import evaluate_news_signal
except Exception:
    evaluate_news_signal = None  # type: ignore

try:
    from app.setup_scorecard import build_setup_scorecard, summarize_scorecard_for_discord
except Exception:
    build_setup_scorecard = None  # type: ignore
    summarize_scorecard_for_discord = None  # type: ignore


# ---------------------------------------------------------------------------
# Sports categories & ticker prefixes
# ---------------------------------------------------------------------------
SPORTS_CATEGORIES: frozenset[str] = frozenset({
    "sports", "sport", "nfl", "nba", "mlb", "nhl", "ncaa", "ncaaf", "ncaab",
    "soccer", "tennis", "golf", "mma", "ufc", "nascar", "formula1", "f1",
    "basketball", "football", "baseball", "hockey", "esports",
    "olympics", "rugby", "cricket", "boxing", "mls", "epl", "fifa",
})

SPORTS_TICKER_PREFIXES: tuple[str, ...] = (
    "NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "NCAAW",
    "MLS", "EPL", "FIFA", "UFC", "MMA", "PGA", "TENNIS",
    "NASCAR", "F1", "KXNFL", "KXNBA", "KXMLB", "KXNHL",
    "KXSOCCER", "KXSPORTS", "KXUFC", "KXMMA", "KXPGA",
)

SPORTS_KEYWORDS: tuple[str, ...] = (
    "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
    "baseball", "hockey", "tennis", "golf", "ufc", "mma", "nascar",
    "super bowl", "superbowl", "world cup", "playoffs", "championship",
    "match winner", "game winner", "season wins", "win total",
    "point spread", "over/under", "moneyline",
)


# ---------------------------------------------------------------------------
# Expert trading constants — aggressive but disciplined
# ---------------------------------------------------------------------------

# Kelly tiers — real quant traders use 25-45%, not 10%
KELLY_TIER_LOW   = 0.18   # edge 2–5%   → 18% of full Kelly
KELLY_TIER_MID   = 0.27   # edge 5–10%  → 27%
KELLY_TIER_HIGH  = 0.35   # edge 10–18% → 35%
KELLY_TIER_ELITE = 0.45   # edge > 18%  → 45% (rare, exceptional plays)

# Stake caps (designed for ~$100 bankroll, scale fast as it grows)
MAX_STAKE_DOLLARS_NORMAL  = 18.0
MAX_STAKE_DOLLARS_STRONG  = 30.0
MAX_STAKE_DOLLARS_ELITE   = 45.0
MIN_STAKE_DOLLARS         = 3.0
MAX_CONTRACTS_NORMAL      = 35
MAX_CONTRACTS_STRONG      = 60
MAX_CONTRACTS_ELITE       = 100
BANKROLL_PCT_CAP          = 0.28   # hard cap: never > 28% of bankroll per trade

# Entry parameters
ENTRY_COOLDOWN_MINUTES     = 5.0   # 5 min cooldown per ticker
FEE_BUFFER                 = 0.004 # 0.4% fee buffer
MAX_SIMULTANEOUS_POSITIONS = 3     # up to 3 open positions

# Elite edge: if edge > this, skip maker and cross spread immediately
ELITE_EDGE_TAKER_THRESHOLD = 0.12

# Taker surcharge over maker threshold (flat, not 2×)
TAKER_EDGE_PREMIUM = 0.020

# Exit parameters
TAKE_PROFIT_NET_CENTS  = 7.0     # harvest after +7 cents
STOP_LOSS_EDGE_FLOOR   = -0.015  # exit if edge drops to -1.5%
STOP_LOSS_MARK_DROP    = -8.0    # emergency exit if mark drops 8 cents against us
MIN_HOLD_MINUTES       = 5.0     # no exit before 5 min (unless emergency)
MAX_HOLD_MINUTES       = 300.0   # 5-hour force exit

# Daily loss cap
DAILY_LOSS_STOP_DOLLARS = -25.0

# Scan top N candidates per tick and pick the one with highest EV
TOP_CANDIDATES_TO_EVALUATE = 5


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
        self._tick_count = 0

        # Paper resolver
        self.paper_resolver_enabled = True
        self.paper_resolver_min_age_minutes = 10
        self.paper_resolver_max_to_check = 30

        # Risk-on gate (lower bar — we want to trade)
        self.min_resolved_for_winrate_gate = 6
        self.min_winrate_for_risk_on_gate  = 0.44

        # News signal config
        self.news_signal_enabled              = True
        self.news_max_abs_score               = 10.0
        self.news_require_two_credible_sources = True
        self.news_min_confidence_to_apply     = 0.30
        self.news_full_confidence_at          = 0.70
        self.news_conflict_penalty_multiplier = 0.40
        self.news_neutral_regime_multiplier   = 0.45
        self.news_mixed_regime_multiplier     = 0.65
        self.news_abs_effect_cap_on_conviction = 8.0
        self.news_allow_direction_flip        = False

        # Thesis-deterioration exit settings
        self.exit_management_enabled        = True
        self.exit_min_hold_minutes          = MIN_HOLD_MINUTES
        self.exit_max_hold_conviction       = 50.0
        self.exit_conviction_drop_points    = 14.0
        self.exit_require_opposite_or_skip  = True
        self.exit_news_conflict_min_conf    = 0.70
        self.exit_news_conflict_min_effect  = 3.0

        # Scorecard cadence
        self.scorecard_summary_every_n_ticks = 20

        # Sports filter
        self.sports_only: bool = True

        # Per-ticker entry cooldowns
        self._last_entry_ts: dict[str, dt.datetime] = {}

        # News cache
        self.news_cache = None
        try:
            import os
            from app.news_cache import NewsCache
            cache_path = os.path.join(os.path.dirname(self.settings.DB_PATH), "news_cache.json")
            self.news_cache = NewsCache(path=cache_path, ttl_seconds=6 * 3600)
        except Exception:
            self.news_cache = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._stopping = True
        for obj in (self.client, self.store):
            try:
                obj.close()
            except Exception:
                pass

    def startup_message(self) -> None:
        msg = (
            f"🚀 Kalshi EXPERT bot | env={self.settings.KALSHI_ENV} "
            f"dry_run={self.settings.DRY_RUN} interval={self.settings.SCAN_INTERVAL_SECONDS}s "
            f"| AGGRESSIVE_EV | max_pos={MAX_SIMULTANEOUS_POSITIONS} "
            f"kelly_elite={KELLY_TIER_ELITE:.0%} max_stake=${MAX_STAKE_DOLLARS_ELITE:.0f} "
            f"take_profit=+{TAKE_PROFIT_NET_CENTS}¢ stop_loss={STOP_LOSS_EDGE_FLOOR:.1%}"
        )
        self.logger.info(msg)
        self.notifier.send(msg)

    # ------------------------------------------------------------------
    # Portfolio snapshot
    # ------------------------------------------------------------------
    def _fetch_portfolio_snapshot(self) -> PortfolioSnapshot:
        snap = PortfolioSnapshot()
        try:
            bal = self.client.get_portfolio_balance()
            snap.cash_balance_dollars = parse_balance_payload(bal)
        except Exception as e:
            self.logger.warning(f"Balance fetch failed: {e}")
        try:
            pos = self.client.get_positions()
            positions = self._extract_positions_list(pos)
            snap.open_positions_count = len(positions)
            snap.open_exposure_dollars = min(
                float(self.settings.MAX_OPEN_EXPOSURE_DOLLARS),
                float(len(positions)) * float(self.settings.MIN_TRADE_DOLLARS),
            )
        except Exception as e:
            self.logger.warning(f"Positions fetch failed: {e}")
        snap.daily_pnl_dollars = 0.0
        return snap

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _unwrap_market(self, raw: Any) -> Optional[dict]:
        if isinstance(raw, dict):
            for k in ("market", "data", "result", "response"):
                v = raw.get(k)
                if isinstance(v, dict):
                    return v
            if "ticker" in raw:
                return raw
        return None

    def _to_float(self, v: Any) -> Optional[float]:
        try:
            if v is None or isinstance(v, bool):
                return None
            return float(v)
        except Exception:
            return None

    def _extract_positions_list(self, payload: Any) -> list[dict]:
        if isinstance(payload, dict):
            for key in ("positions", "market_positions", "event_positions"):
                p = payload.get(key)
                if isinstance(p, list):
                    return [x for x in p if isinstance(x, dict)]
            for outer in ("data", "result", "response"):
                nested = payload.get(outer)
                if isinstance(nested, dict):
                    for key in ("positions", "market_positions", "event_positions"):
                        p = nested.get(key)
                        if isinstance(p, list):
                            return [x for x in p if isinstance(x, dict)]
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        return []

    def _position_ticker(self, p: dict) -> str:
        for k in ("ticker", "market_ticker", "event_ticker", "instrument_ticker"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _position_contract_side(self, p: dict) -> Optional[str]:
        for k in ("yes_no", "side", "position_side", "contract_side", "outcome"):
            v = p.get(k)
            if v is None:
                continue
            s = str(v).strip().upper()
            if s in {"YES", "Y"}:
                return "YES"
            if s in {"NO", "N"}:
                return "NO"
        yes_qty = self._to_float(p.get("yes_count") or p.get("yes_position") or p.get("yes_contracts"))
        no_qty  = self._to_float(p.get("no_count")  or p.get("no_position")  or p.get("no_contracts"))
        if yes_qty and yes_qty > 0 and (not no_qty or no_qty <= 0):
            return "YES"
        if no_qty and no_qty > 0 and (not yes_qty or yes_qty <= 0):
            return "NO"
        return None

    def _position_count(self, p: dict) -> int:
        for k in ("count", "contracts", "qty", "quantity", "position", "open_interest"):
            v = self._to_float(p.get(k))
            if v is not None and v > 0:
                return max(0, int(round(v)))
        for k in ("yes_count", "yes_contracts", "yes_position", "no_count", "no_contracts", "no_position"):
            v = self._to_float(p.get(k))
            if v is not None and v > 0:
                return max(0, int(round(v)))
        return 0

    def _position_entry_price_cents(self, p: dict) -> Optional[int]:
        for k in ("avg_price_cents", "average_price_cents", "entry_price_cents", "cost_basis_price_cents"):
            v = self._to_float(p.get(k))
            if v is not None:
                return int(round(v))
        return None

    def _position_open_ts(self, p: dict) -> Optional[dt.datetime]:
        for k in ("opened_ts", "opened_at", "created_ts", "created_at", "updated_at", "ts"):
            raw = p.get(k)
            if not raw:
                continue
            try:
                s = str(raw).replace("Z", "+00:00")
                d = dt.datetime.fromisoformat(s)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=dt.UTC)
                return d.astimezone(dt.UTC)
            except Exception:
                continue
        return None

    def _minutes_since(self, ts_like: Any) -> Optional[float]:
        if ts_like is None:
            return None
        try:
            s = str(ts_like).replace("Z", "+00:00")
            d = dt.datetime.fromisoformat(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.UTC)
            return max(0.0, (dt.datetime.now(dt.UTC) - d.astimezone(dt.UTC)).total_seconds() / 60.0)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Sports filter
    # ------------------------------------------------------------------
    def _is_sports_market(self, m: dict) -> bool:
        cat = str(m.get("category", "") or "").strip().lower()
        if cat and cat in SPORTS_CATEGORIES:
            return True
        ticker = str(m.get("ticker", "") or "").upper().strip()
        if any(ticker.startswith(p) for p in SPORTS_TICKER_PREFIXES):
            return True
        for field in ("title", "question", "market_label", "name"):
            text = str(m.get(field, "") or "").lower()
            if any(kw in text for kw in SPORTS_KEYWORDS):
                return True
        # Unknown category: allow only if liquidity is exceptional
        if not cat:
            spread = float(m.get("spread_cents", 999) or 999)
            vol    = float(m.get("volume", 0) or 0)
            if spread <= 2.0 and vol >= 1000.0:
                return True
        return False

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    def _is_in_cooldown(self, ticker: str) -> bool:
        last = self._last_entry_ts.get(ticker)
        if last is None:
            return False
        return (dt.datetime.now(dt.UTC) - last).total_seconds() / 60.0 < ENTRY_COOLDOWN_MINUTES

    def _set_entry_cooldown(self, ticker: str) -> None:
        self._last_entry_ts[ticker] = dt.datetime.now(dt.UTC)

    # ------------------------------------------------------------------
    # Daily loss cap
    # ------------------------------------------------------------------
    def _compute_daily_realized_pnl(self) -> float:
        today = dt.datetime.now(dt.UTC).date()
        try:
            rows = self.store.recent_decisions(200)
        except Exception:
            return 0.0
        total = 0.0
        for r in rows:
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            if not r.get("resolved_ts"):
                continue
            try:
                s = str(r["resolved_ts"]).replace("Z", "+00:00")
                d = dt.datetime.fromisoformat(s)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=dt.UTC)
                if d.date() != today:
                    continue
            except Exception:
                continue
            pnl = self._to_float(r.get("realized_pnl"))
            if pnl is not None:
                total += pnl
        return total

    def _daily_loss_cap_breached(self) -> bool:
        daily_pnl = self._compute_daily_realized_pnl()
        if daily_pnl <= DAILY_LOSS_STOP_DOLLARS:
            self.logger.warning(f"Daily loss cap breached: {daily_pnl:.2f} <= {DAILY_LOSS_STOP_DOLLARS:.2f}")
            return True
        return False

    # ------------------------------------------------------------------
    # Synthetic open positions (DRY_RUN)
    # ------------------------------------------------------------------
    def _synthetic_open_positions_from_decisions(self) -> list[dict]:
        if not bool(self.settings.DRY_RUN):
            return []
        try:
            rows = self.store.recent_decisions(1000)
        except Exception:
            return []

        open_pos: dict[str, dict] = {}
        for r in reversed(rows):
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            ticker  = str(r.get("ticker", "") or "").strip()
            if not ticker:
                continue
            action  = str(r.get("action", "") or "")
            reasons = str(r.get("reasons", "") or "")

            if action.startswith("TRADE_EXIT"):
                exit_finalized = (
                    bool(r.get("resolved_ts"))
                    or "exit_consumed:1" in reasons
                    or "exit_resolved:1" in reasons
                )
                if ticker in open_pos:
                    if exit_finalized:
                        del open_pos[ticker]
                    else:
                        open_pos[ticker]["_exit_pending"] = True
                continue

            if action in {"TRADE_YES", "TRADE_NO"}:
                if r.get("resolved_ts") or "mtm_label:" in reasons:
                    open_pos.pop(ticker, None)
                    continue

                entry_px: Optional[int] = None
                count: Optional[int]    = None
                for part in reasons.split(";"):
                    part = part.strip()
                    if part.startswith("entry_price_cents:"):
                        try:
                            entry_px = int(float(part.split(":", 1)[1]))
                        except Exception:
                            pass
                    elif part.startswith("order_count:"):
                        try:
                            count = int(float(part.split(":", 1)[1]))
                        except Exception:
                            pass

                if not count or count <= 0:
                    stake = self._to_float(r.get("stake_dollars")) or 0.0
                    if entry_px and entry_px > 0:
                        try:
                            count = max(1, int(round((stake * 100.0) / float(entry_px))))
                        except Exception:
                            count = 1
                    else:
                        count = 1

                entry_id = r.get("id")
                if entry_id is not None:
                    try:
                        entry_id = int(entry_id)
                    except Exception:
                        entry_id = None

                open_pos[ticker] = {
                    "ticker": ticker,
                    "yes_no": "YES" if action == "TRADE_YES" else "NO",
                    "count": max(1, int(count)),
                    "entry_price_cents": entry_px,
                    "opened_at": r.get("ts"),
                    "entry_id": entry_id,
                    "_synthetic_from_decisions": True,
                    "_exit_pending": False,
                }

        return list(open_pos.values())

    def _synthetic_open_position_from_decisions(self, ticker: str) -> Optional[dict]:
        for pos in self._synthetic_open_positions_from_decisions():
            if str(pos.get("ticker")) == str(ticker or "").strip():
                return pos
        return None

    def _find_open_position_for_ticker(self, ticker: str) -> Optional[dict]:
        t = str(ticker or "").strip()
        if not t:
            return None
        try:
            raw = self.client.get_positions()
            for p in self._extract_positions_list(raw):
                if self._position_ticker(p) == t:
                    if self._position_count(p) > 0 and self._position_contract_side(p) in {"YES", "NO"}:
                        return p
        except Exception as e:
            self.logger.warning(f"Position scan failed: {e}")
        return self._synthetic_open_position_from_decisions(t)

    def _all_open_position_tickers(self) -> set[str]:
        synth = {str(s.get("ticker", "")) for s in self._synthetic_open_positions_from_decisions()}
        try:
            raw = self.client.get_positions()
            live = {
                self._position_ticker(p)
                for p in self._extract_positions_list(raw)
                if self._position_count(p) > 0 and self._position_contract_side(p)
            }
        except Exception:
            live = set()
        return synth | live

    def _latest_unresolved_entry_id_for_ticker(self, ticker: str) -> Optional[int]:
        t = str(ticker or "").strip()
        if not t:
            return None
        try:
            rows = self.store.recent_decisions(250)
        except Exception:
            return None
        for r in rows:
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            if str(r.get("ticker", "")) != t:
                continue
            if str(r.get("action", "")) not in {"TRADE_YES", "TRADE_NO"}:
                continue
            reasons = str(r.get("reasons", "") or "")
            if r.get("resolved_ts") or "mtm_label:" in reasons:
                continue
            entry_id = r.get("id")
            if entry_id is not None:
                try:
                    return int(entry_id)
                except Exception:
                    pass
        return None

    def _latest_entry_context_for_ticker(self, ticker: str) -> dict:
        out: dict = {"entry_conviction": None, "entry_ts": None,
                     "entry_price_cents": None, "entry_action": None}
        t = str(ticker or "").strip()
        if not t:
            return out
        try:
            rows = self.store.recent_decisions(250)
        except Exception:
            return out
        for r in rows:
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            if str(r.get("ticker", "")) != t:
                continue
            action = str(r.get("action", ""))
            if not action.startswith("TRADE_") or action.startswith("TRADE_EXIT"):
                continue
            out["entry_conviction"] = self._to_float(r.get("final_conviction"))
            out["entry_ts"]         = r.get("ts")
            out["entry_action"]     = action
            reasons = str(r.get("reasons", "") or "")
            idx = reasons.find("entry_price_cents:")
            if idx >= 0:
                tail = reasons[idx + len("entry_price_cents:"):]
                try:
                    out["entry_price_cents"] = int(float(tail.split(";", 1)[0].strip()))
                except Exception:
                    pass
            return out
        return out

    # ------------------------------------------------------------------
    # EV gate — the single source of truth for every trade decision
    # ------------------------------------------------------------------
    def _compute_ev_gate(
        self,
        direction: str,
        final_conviction: float,
        chosen: dict,
        limit_price_cents: int,
        exec_mode: str,
    ) -> dict:
        """
        p_model  = final_conviction / 100
        p_market = limit_price / 100
        edge     = p_model - p_market

        cost_buffer = max(0.01, spread * 0.5%) + FEE_BUFFER
        Maker threshold: edge > cost_buffer
        Taker threshold: edge > cost_buffer + TAKER_EDGE_PREMIUM (flat, not 2×)
        """
        p_model  = max(0.01, min(0.99, final_conviction / 100.0))
        p_market = max(0.01, min(0.99, limit_price_cents / 100.0))
        edge     = p_model - p_market

        spread_cents = float(chosen.get("spread_cents", 0) or 0)
        volume       = float(chosen.get("volume", 0) or 0)
        cost_buffer  = max(0.01, (spread_cents / 100.0) * 0.50) + FEE_BUFFER
        threshold    = cost_buffer + TAKER_EDGE_PREMIUM if exec_mode == "taker" else cost_buffer

        spread_ok = spread_cents <= float(self.settings.MAX_SPREAD_CENTS)
        volume_ok = volume >= float(self.settings.MIN_RECENT_VOLUME)
        ev_pass   = (edge > threshold) and spread_ok and volume_ok

        tier = ("elite" if edge > 0.18 else "high" if edge > 0.10
                else "mid" if edge > 0.05 else "low" if edge > 0.02 else "subthreshold")

        return {
            "p_model": round(p_model, 4),
            "p_market": round(p_market, 4),
            "edge": round(edge, 4),
            "cost_buffer": round(cost_buffer, 4),
            "threshold": round(threshold, 4),
            "spread_ok": spread_ok,
            "volume_ok": volume_ok,
            "ev_pass": ev_pass,
            "exec_mode": exec_mode,
            "tier": tier,
            "spread_cents": spread_cents,
            "volume": volume,
        }

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def _compute_maker_entry_price(self, direction: str, chosen: dict) -> tuple[int, str]:
        yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
        yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
        mid     = self._to_float(chosen.get("mid_yes_cents")) or 50.0

        if direction == "YES":
            px = min(yes_ask - 1, mid + 0.5) if yes_ask > 1 else max(1.0, mid - 1)
        else:
            no_ask = 100.0 - yes_bid if yes_bid > 0 else 99.0
            no_bid = 100.0 - yes_ask if yes_ask > 0 else 1.0
            no_mid = (no_ask + no_bid) / 2.0 if no_ask > no_bid else (100.0 - mid)
            px = min(no_ask - 1, no_mid + 0.5) if no_ask > 1 else max(1.0, no_mid - 1)
        return int(max(1, min(99, round(px)))), "maker"

    def _taker_price(self, direction: str, chosen: dict) -> int:
        yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
        yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
        mid     = self._to_float(chosen.get("mid_yes_cents")) or 50.0
        if direction == "YES":
            px = yes_ask if yes_ask > 0 else round(mid + 1)
        else:
            px = (100.0 - yes_bid) if yes_bid > 0 else 99.0
        return int(max(1, min(99, round(px))))

    def _best_entry_price_and_mode(
        self, direction: str, final_conviction: float, chosen: dict,
    ) -> tuple[int, str, dict]:
        maker_price, _ = self._compute_maker_entry_price(direction, chosen)
        ev_maker = self._compute_ev_gate(direction, final_conviction, chosen, maker_price, "maker")

        # Elite edge: cross the spread — don't risk missing the fill
        if ev_maker["edge"] >= ELITE_EDGE_TAKER_THRESHOLD:
            taker_price = self._taker_price(direction, chosen)
            ev_taker    = self._compute_ev_gate(direction, final_conviction, chosen, taker_price, "taker")
            if ev_taker["ev_pass"]:
                return taker_price, "taker", ev_taker

        if ev_maker["ev_pass"]:
            return maker_price, "maker", ev_maker

        # Fall back to taker
        taker_price = self._taker_price(direction, chosen)
        ev_taker    = self._compute_ev_gate(direction, final_conviction, chosen, taker_price, "taker")
        if ev_taker["ev_pass"]:
            return taker_price, "taker", ev_taker

        return maker_price, "maker", ev_maker  # neither passed — caller checks ev_pass

    # ------------------------------------------------------------------
    # Expert Kelly sizing — tiered by edge quality
    # ------------------------------------------------------------------
    def _compute_expert_kelly_stake(self, ev: dict, bankroll_dollars: float) -> tuple[float, int]:
        """
        Returns (stake_dollars, contracts).
        Uses tiered Kelly damping based on edge quality.
        Larger edges → larger fraction of Kelly → larger positions.
        """
        p_market = max(0.01, min(0.99, ev["p_market"]))
        p_model  = max(0.01, min(0.99, ev["p_model"]))
        edge     = ev["edge"]
        tier     = ev.get("tier", "low")
        limit_px = int(round(p_market * 100.0))

        full_kelly = max(0.0, edge / max(0.01, 1.0 - p_market))

        if tier == "elite":
            damping, max_stake, max_ct = KELLY_TIER_ELITE, MAX_STAKE_DOLLARS_ELITE, MAX_CONTRACTS_ELITE
        elif tier == "high":
            damping, max_stake, max_ct = KELLY_TIER_HIGH,  MAX_STAKE_DOLLARS_STRONG, MAX_CONTRACTS_STRONG
        elif tier == "mid":
            damping, max_stake, max_ct = KELLY_TIER_MID,   MAX_STAKE_DOLLARS_NORMAL, MAX_CONTRACTS_NORMAL
        else:
            damping, max_stake, max_ct = KELLY_TIER_LOW,   MAX_STAKE_DOLLARS_NORMAL, MAX_CONTRACTS_NORMAL

        # High model confidence gets a boost
        if p_model >= 0.80:
            damping = min(damping * 1.30, KELLY_TIER_ELITE)

        raw_stake = bankroll_dollars * full_kelly * damping
        stake     = min(max_stake, raw_stake)
        stake     = min(stake, bankroll_dollars * BANKROLL_PCT_CAP)
        stake     = max(MIN_STAKE_DOLLARS, stake)

        contracts = max(1, min(max_ct, int(round((stake * 100.0) / max(1, limit_px)))))
        actual    = round((contracts * limit_px) / 100.0, 2)
        return actual, contracts

    # ------------------------------------------------------------------
    # Multi-candidate EV selection: pick the highest-edge opportunity
    # ------------------------------------------------------------------
    def _select_best_ev_candidate(
        self,
        candidates: list[dict],
        final_convictions: dict[str, float],
    ) -> Optional[tuple[dict, str, int, str, dict]]:
        best: Optional[tuple[dict, str, int, str, dict]] = None
        best_edge = -999.0

        for c in candidates:
            ticker  = str(c.get("ticker", ""))
            if not ticker:
                continue
            sig_dir = str(c.get("_signal_direction", "SKIP") or "SKIP").upper()
            if sig_dir == "SKIP":
                continue
            conv    = final_convictions.get(ticker, 0.0)
            price, mode, ev = self._best_entry_price_and_mode(sig_dir, conv, c)
            if not ev["ev_pass"]:
                continue
            if ev["edge"] > best_edge:
                best_edge = ev["edge"]
                best = (c, sig_dir, price, mode, ev)

        return best

    # ------------------------------------------------------------------
    # Exit: profit / stoploss / max-hold
    # ------------------------------------------------------------------
    def _should_exit_profit_or_stoploss(
        self, *, chosen: dict, pos: dict, final_conviction: float, ev: dict,
    ) -> tuple[bool, str, list[str]]:
        side  = self._position_contract_side(pos)
        count = self._position_count(pos)
        if side not in {"YES", "NO"} or count <= 0:
            return False, "no_position", []

        opened_ts = self._position_open_ts(pos)
        if opened_ts is not None:
            held_minutes = max(0.0, (dt.datetime.now(dt.UTC) - opened_ts).total_seconds() / 60.0)
        else:
            ctx = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
            m   = self._minutes_since(ctx.get("entry_ts"))
            held_minutes = m if m is not None else 0.0

        reasons: list[str] = [f"exit_held:{side}", f"exit_mins:{held_minutes:.1f}"]

        # 5-hour force exit
        if held_minutes >= MAX_HOLD_MINUTES:
            reasons.append(f"exit_max_hold:{MAX_HOLD_MINUTES:.0f}m")
            return True, "max_hold", reasons

        entry_px = self._position_entry_price_cents(pos)
        if entry_px is None:
            ctx = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
            entry_px = ctx.get("entry_price_cents")

        edge = ev.get("edge", 0.0)

        if entry_px is not None:
            yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
            yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
            mid     = self._to_float(chosen.get("mid_yes_cents")) or 50.0

            if side == "YES":
                current_mark = yes_bid if yes_bid > 0 else mid
            else:
                current_mark = max(1.0, 100.0 - yes_ask) if yes_ask > 0 else (100.0 - mid)

            mark_move = current_mark - float(entry_px)
            reasons.extend([f"exit_entry_px:{entry_px}",
                             f"exit_mark:{current_mark:.1f}",
                             f"exit_move:{mark_move:.2f}",
                             f"exit_edge:{edge:.4f}"])

            # Emergency: mark dropped hard AND edge turned negative → bypass min hold
            emergency = (mark_move <= STOP_LOSS_MARK_DROP and edge < STOP_LOSS_EDGE_FLOOR)
            if emergency:
                reasons.append(f"exit_emergency:move={mark_move:.2f}_edge={edge:.4f}")
                return True, "emergency_stop", reasons

            # Min hold guard (skip if emergency already handled above)
            if held_minutes < MIN_HOLD_MINUTES:
                reasons.append("exit_min_hold_guard")
                return False, "hold_guard", reasons

            # Take profit
            if mark_move >= TAKE_PROFIT_NET_CENTS:
                reasons.append(f"exit_take_profit:+{mark_move:.2f}¢")
                return True, "take_profit", reasons

            # Stop loss: edge collapsed
            if edge < STOP_LOSS_EDGE_FLOOR:
                reasons.append(f"exit_stop_loss:edge={edge:.4f}")
                return True, "stop_loss", reasons

            # Liquidity deterioration while underwater
            spread = float(chosen.get("spread_cents", 0) or 0)
            if spread > float(self.settings.MAX_SPREAD_CENTS) * 2.5 and mark_move < -3.0:
                reasons.append(f"exit_spread_blown:{spread:.1f}")
                return True, "spread_blown", reasons

        elif held_minutes >= MIN_HOLD_MINUTES:
            reasons.append("exit_no_entry_px")

        return False, "hold", reasons

    # ------------------------------------------------------------------
    # Exit: thesis deterioration
    # ------------------------------------------------------------------
    def _should_exit_on_thesis_deterioration(
        self, *, chosen: dict, sig: Any, final_conviction: float,
        news: dict, news_applied: dict,
    ) -> tuple[bool, list[str]]:
        if not self.exit_management_enabled:
            return False, ["exit_mgmt_disabled"]

        pos = self._find_open_position_for_ticker(str(chosen.get("ticker")))
        if not pos:
            return False, ["no_open_position"]

        held_side  = self._position_contract_side(pos)
        held_count = self._position_count(pos)
        if held_side not in {"YES", "NO"} or held_count <= 0:
            return False, ["position_unparseable"]

        p_opened = self._position_open_ts(pos)
        if p_opened is not None:
            held_minutes: Optional[float] = max(0.0, (dt.datetime.now(dt.UTC) - p_opened).total_seconds() / 60.0)
        else:
            entry_ctx = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
            held_minutes = self._minutes_since(entry_ctx.get("entry_ts"))

        reasons: list[str] = [f"exit_thesis:{held_side}", f"exit_thesis_mins:{(held_minutes or 0):.1f}"]

        if held_minutes is not None and held_minutes < float(self.exit_min_hold_minutes):
            reasons.append("exit_thesis_hold_guard")
            return False, reasons

        sig_dir    = str(getattr(sig, "direction", "SKIP") or "SKIP").upper()
        final_conv = float(final_conviction)

        entry_ctx  = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
        entry_conv = self._to_float(entry_ctx.get("entry_conviction"))
        if entry_conv is not None:
            reasons.append(f"exit_thesis_entry_conv:{entry_conv:.1f}")
            reasons.append(f"exit_thesis_conv_drop:{(entry_conv - final_conv):.1f}")

        opposite_signal      = (held_side == "YES" and sig_dir == "NO") or (held_side == "NO" and sig_dir == "YES")
        weak_or_skip         = sig_dir == "SKIP" and final_conv <= float(self.exit_max_hold_conviction)
        conviction_collapsed = entry_conv is not None and (entry_conv - final_conv) >= float(self.exit_conviction_drop_points)

        news_conf = float(news.get("confidence", 0.0) or 0.0)
        eff_news  = float(news_applied.get("effective_news_score", 0.0) or 0.0)
        strong_adverse_news = (
            (held_side == "YES" and eff_news <= -float(self.exit_news_conflict_min_effect))
            or (held_side == "NO" and eff_news >= float(self.exit_news_conflict_min_effect))
        ) and news_conf >= float(self.exit_news_conflict_min_conf)

        reasons.extend([
            f"exit_thesis_sig_dir:{sig_dir}",
            f"exit_thesis_final_conv:{final_conv:.1f}",
            f"exit_thesis_opp:{1 if opposite_signal else 0}",
            f"exit_thesis_weak:{1 if weak_or_skip else 0}",
            f"exit_thesis_collapse:{1 if conviction_collapsed else 0}",
            f"exit_thesis_adverse_news:{1 if strong_adverse_news else 0}",
        ])

        if self.exit_require_opposite_or_skip:
            if not (opposite_signal or weak_or_skip):
                return False, reasons
            if not (opposite_signal or conviction_collapsed or strong_adverse_news):
                return False, reasons
        else:
            if not (opposite_signal or weak_or_skip or conviction_collapsed or strong_adverse_news):
                return False, reasons

        return True, reasons

    # ------------------------------------------------------------------
    # Exit order builder (maker-first, aggressive)
    # ------------------------------------------------------------------
    def _build_exit_order(self, chosen: dict, pos: dict) -> Optional[dict]:
        side  = self._position_contract_side(pos)
        count = self._position_count(pos)
        if side not in {"YES", "NO"} or count <= 0:
            return None

        yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
        yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
        no_bid  = self._to_float(chosen.get("no_bid")) or 0.0
        mid     = self._to_float(chosen.get("mid_yes_cents")) or 50.0

        if side == "YES":
            px     = int(min(99, max(1, round((yes_bid + 1) if yes_bid > 0 else mid))))
            yes_no = "yes"
        else:
            if no_bid > 0:
                px = int(min(99, max(1, round(no_bid + 1))))
            elif yes_ask > 0:
                px = int(max(1, min(99, round(100 - yes_ask + 1))))
            else:
                px = int(max(1, min(99, round(100.0 - mid))))
            yes_no = "no"

        return {
            "ticker": str(chosen.get("ticker", "")),
            "side": "sell",
            "yes_no": yes_no,
            "count": int(count),
            "price_cents": int(max(1, min(99, px))),
            "client_order_id": f"exit-{chosen.get('ticker')}-{int(dt.datetime.now(dt.UTC).timestamp())}",
            "held_side": side,
        }

    # ------------------------------------------------------------------
    # Decision recording (backward compat)
    # ------------------------------------------------------------------
    def _record_decision_row(self, row: dict) -> Optional[int]:
        inserted_id: Optional[int] = None
        try:
            inserted_id = self.store.insert_decision(row)
        except TypeError:
            legacy_row = {k: row.get(k) for k in (
                "ts", "ticker", "direction", "base_conviction", "social_bonus",
                "final_conviction", "stake_dollars", "action", "reasons",
                "dry_run", "realized_pnl", "won", "resolved_ts", "market_category",
            )}
            inserted_id = self.store.insert_decision(legacy_row)
            try:
                if hasattr(self.store, "update_decision_metadata") and inserted_id is not None:
                    self.store.update_decision_metadata(
                        decision_id=int(inserted_id),
                        updates={k: row.get(k) for k in (
                            "news_score", "news_confidence", "news_regime",
                            "news_effective_score", "spread_cents",
                        )},
                    )
            except Exception as e:
                self.logger.warning(f"Post-insert metadata patch failed: {e}")
        except Exception as e:
            self.logger.warning(f"Decision insert failed: {e}")
        return inserted_id

    # ------------------------------------------------------------------
    # News signal plumbing
    # ------------------------------------------------------------------
    def _news_category_allowlist(self) -> list[str]:
        return ["politics", "weather", "sports", "macro"]

    def _fetch_news_items_for_market(self, chosen: dict) -> list[dict]:
        try:
            from app.news_fetch import (
                build_news_query_from_market_label, fetch_google_news_rss,
                infer_direction_hint, yes_means_event_occurs,
            )
        except Exception as e:
            self.logger.warning(f"News fetch module unavailable: {e}")
            return []

        label = str(
            chosen.get("market_label") or chosen.get("title")
            or chosen.get("question") or chosen.get("name") or chosen.get("ticker") or ""
        ).strip()
        if not label:
            return []
        query = build_news_query_from_market_label(label)
        if not query:
            return []

        self.logger.info(f"News query='{query}' ticker={chosen.get('ticker')}")
        cache     = getattr(self, "news_cache", None)
        raw_items = fetch_google_news_rss(query, max_items=12, timeout=6.0)
        if not raw_items:
            return []

        y_is_event = yes_means_event_occurs(label)
        out: list[dict] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            if cache is not None:
                try:
                    if cache.seen_recently(it):
                        continue
                except Exception:
                    pass
            try:
                enriched = infer_direction_hint(it, yes_means_event_occurs=y_is_event)
            except Exception:
                enriched = it
            out.append(enriched)
            if cache is not None:
                try:
                    cache.mark_seen(it)
                except Exception:
                    pass
        return out

    def _compute_news_signal(self, chosen: dict) -> dict:
        _empty = {
            "score": 0.0, "confidence": 0.0, "regime": "disabled",
            "reasons": ["news_disabled"], "risk_flags": [],
            "n_items": 0, "n_unique": 0, "n_tier1": 0, "n_tier2": 0, "n_tier3": 0,
        }
        if not self.news_signal_enabled or evaluate_news_signal is None:
            return _empty
        try:
            news_items = self._fetch_news_items_for_market(chosen)
        except Exception as e:
            self.logger.warning(f"News fetch failed: {e}")
            news_items = []
        try:
            ns = evaluate_news_signal(
                chosen, news_items, enabled=True,
                max_abs_score=self.news_max_abs_score,
                require_two_credible_sources_for_boost=self.news_require_two_credible_sources,
                category_allowlist=self._news_category_allowlist(),
            )
            return {
                "score": float(getattr(ns, "score", 0.0)),
                "confidence": float(getattr(ns, "confidence", 0.0)),
                "regime": str(getattr(ns, "regime", "unavailable")),
                "reasons": list(getattr(ns, "reasons", [])),
                "risk_flags": list(getattr(ns, "risk_flags", [])),
                "n_items": int(getattr(ns, "n_items", 0)),
                "n_unique": int(getattr(ns, "n_unique", 0)),
                "n_tier1": int(getattr(ns, "n_tier1", 0)),
                "n_tier2": int(getattr(ns, "n_tier2", 0)),
                "n_tier3": int(getattr(ns, "n_tier3", 0)),
            }
        except Exception as e:
            self.logger.warning(f"News signal failed: {e}")
            return {**_empty, "regime": "error_fallback", "reasons": ["news_eval_error"]}

    def _apply_news_to_conviction(
        self, *, base_signal_direction: str, base_conviction: float,
        social_bonus: float, news: dict,
    ) -> tuple[float, dict]:
        raw_news_score = float(news.get("score", 0.0))
        news_conf      = max(0.0, min(1.0, float(news.get("confidence", 0.0))))
        news_regime    = str(news.get("regime", "unavailable"))
        sig_dir        = str(base_signal_direction or "SKIP").upper()

        if news_conf <= self.news_min_confidence_to_apply:
            conf_scale = 0.10
        elif news_conf >= self.news_full_confidence_at:
            conf_scale = 1.00
        else:
            span = max(1e-9, self.news_full_confidence_at - self.news_min_confidence_to_apply)
            conf_scale = 0.10 + 0.90 * ((news_conf - self.news_min_confidence_to_apply) / span)

        regime_scale = 1.0
        if news_regime in {"noisy_or_weak", "unavailable", "disabled", "error_fallback"}:
            regime_scale = self.news_neutral_regime_multiplier
        elif news_regime == "mixed":
            regime_scale = self.news_mixed_regime_multiplier

        conflict        = (sig_dir == "YES" and raw_news_score < 0) or (sig_dir == "NO" and raw_news_score > 0)
        direction_scale = self.news_conflict_penalty_multiplier if conflict else 1.0
        eff             = raw_news_score * conf_scale * regime_scale * direction_scale

        if not self.news_allow_direction_flip and conflict:
            eff = max(-2.0, min(2.0, eff))
        eff = max(-self.news_abs_effect_cap_on_conviction, min(self.news_abs_effect_cap_on_conviction, eff))

        final_conviction = min(100.0, max(0.0, base_conviction + social_bonus + eff))
        return final_conviction, {
            "raw_news_score": round(raw_news_score, 3),
            "effective_news_score": round(eff, 3),
            "news_confidence": round(news_conf, 3),
            "news_regime": news_regime,
            "news_conf_scale": round(conf_scale, 3),
            "news_regime_scale": round(regime_scale, 3),
            "news_direction_scale": round(direction_scale, 3),
            "news_conflicts_signal": conflict,
        }

    # ------------------------------------------------------------------
    # Learning & stats
    # ------------------------------------------------------------------
    def _compute_learning(self, recent: list[dict]) -> dict:
        if compute_learning_adjustment is None:
            return {"stake_multiplier": 1.0, "conviction_adjustment": 0.0,
                    "mode": "disabled", "reasons": ["learning_module_missing"]}
        try:
            adj = compute_learning_adjustment(recent)
            return {
                "stake_multiplier": float(getattr(adj, "stake_multiplier", 1.0)),
                "conviction_adjustment": float(getattr(adj, "conviction_adjustment", 0.0)),
                "mode": str(getattr(adj, "mode", "neutral")),
                "reasons": list(getattr(adj, "reasons", [])),
            }
        except Exception as e:
            self.logger.warning(f"Learning failed: {e}")
            return {"stake_multiplier": 1.0, "conviction_adjustment": 0.0,
                    "mode": "error_fallback", "reasons": ["learning_error"]}

    def _mode_stats_from_decisions(self, decisions: list[dict]) -> dict:
        def summarize_mode(rows: list[dict]) -> dict:
            total    = len(rows)
            trades   = [r for r in rows if str(r.get("action", "")).startswith("TRADE")]
            skips    = [r for r in rows if r.get("action") == "SKIP"]
            errors   = [r for r in rows if r.get("action") == "ERROR"]
            resolved = [r for r in trades if r.get("resolved_ts") is not None]
            pnl_vals = [r.get("realized_pnl") for r in resolved if r.get("realized_pnl") is not None]
            wins     = [r for r in resolved if r.get("won") is True or r.get("won") == 1]
            losses   = [r for r in resolved if r.get("won") is False or r.get("won") == 0]
            avg_conv  = round(sum(float(r.get("final_conviction", 0.0)) for r in rows) / total, 2) if total else 0.0
            avg_stake = round(sum(float(r.get("stake_dollars", 0.0)) for r in trades) / len(trades), 2) if trades else 0.0
            pnl_total = round(sum(float(x) for x in pnl_vals), 2) if pnl_vals else None
            win_rate  = round(len(wins) / (len(wins) + len(losses)) * 100.0, 1) if (wins or losses) else None
            expectancy = round(sum(float(x) for x in pnl_vals) / len(pnl_vals), 4) if pnl_vals else None
            return {
                "count": total, "trades": len(trades), "skips": len(skips), "errors": len(errors),
                "resolved_trades": len(resolved), "avg_conviction": avg_conv, "avg_stake": avg_stake,
                "wins": len(wins), "losses": len(losses), "win_rate": win_rate,
                "pnl_total": pnl_total, "expectancy": expectancy,
            }
        return {
            "sim":  summarize_mode([d for d in decisions if bool(d.get("dry_run", True))]),
            "live": summarize_mode([d for d in decisions if not bool(d.get("dry_run", True))]),
        }

    def _compute_recent_winrate_for_risk_gate(
        self, decisions: list[dict], *, use_dry_run_mode: bool, lookback_resolved: int = 12,
    ) -> tuple[Optional[float], int]:
        mode_rows = [d for d in decisions if bool(d.get("dry_run", True)) == bool(use_dry_run_mode)]
        resolved  = [d for d in mode_rows
                     if str(d.get("action", "")).startswith("TRADE") and d.get("resolved_ts")]
        wins = losses = counted = 0
        for d in resolved[:lookback_resolved]:
            w = d.get("won")
            if w is True or w == 1:
                wins += 1; counted += 1
            elif w is False or w == 0:
                losses += 1; counted += 1
        if counted < self.min_resolved_for_winrate_gate:
            return None, counted
        return (wins / counted) if counted > 0 else None, counted

    # ------------------------------------------------------------------
    # Scorecard / periodic summary
    # ------------------------------------------------------------------
    def _maybe_send_scorecard_summary(self) -> None:
        if build_setup_scorecard is None or summarize_scorecard_for_discord is None:
            return
        if self.scorecard_summary_every_n_ticks <= 0:
            return
        if (self._tick_count % self.scorecard_summary_every_n_ticks) != 0:
            return
        try:
            decisions = self.store.recent_decisions(250)
            scorecard = build_setup_scorecard(decisions)
            if scorecard:
                self.notifier.send(summarize_scorecard_for_discord(scorecard, top_n=3))
        except Exception as e:
            self.logger.warning(f"Scorecard summary failed: {e}")

    def _maybe_send_periodic_summary(self, snapshot: PortfolioSnapshot, recent: list[dict]) -> None:
        if not recent or (self._tick_count % 10) != 0:
            return
        ms = self._mode_stats_from_decisions(recent)

        def _fmt(name: str, m: dict) -> str:
            wl  = f"W/L {m['wins']}-{m['losses']} ({m['win_rate']}%)" if m["win_rate"] is not None else "W/L n/a"
            pnl = f"PnL ${m['pnl_total']:.2f}" if m["pnl_total"] is not None else "PnL n/a"
            exp = f"Exp ${m['expectancy']:.4f}" if m["expectancy"] is not None else "Exp n/a"
            return (f"{name}: trades={m['trades']} resolved={m['resolved_trades']} "
                    f"avg_conv={m['avg_conviction']:.1f} avg_stake=${m['avg_stake']:.2f} {wl} {pnl} {exp}")

        self.notifier.send(
            f"📊 Expert Bot | DRY_RUN={self.settings.DRY_RUN} "
            f"cash≈${snapshot.cash_balance_dollars:.2f} exp≈${snapshot.open_exposure_dollars:.2f}\n"
            f"{_fmt('SIM', ms['sim'])}\n{_fmt('LIVE', ms['live'])}"
        )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def _post_tick_housekeeping(self, snapshot: PortfolioSnapshot) -> None:
        if self.settings.DRY_RUN and self.paper_resolver_enabled:
            try:
                rr = resolve_paper_trades(
                    store=self.store, client=self.client, logger=self.logger,
                    max_to_check=self.paper_resolver_max_to_check,
                    min_age_minutes=self.paper_resolver_min_age_minutes,
                )
                if getattr(rr, "updated", 0) > 0:
                    self.logger.info(f"Paper resolver updated={rr.updated} checked={rr.checked}")
            except Exception as e:
                self.logger.warning(f"Paper resolver failed: {e}")

        try:
            recent20 = self.store.recent_decisions(20)
            self.logger.info(f"Recent summary: {summarize_decisions(recent20)}")
        except Exception as e:
            self.logger.warning(f"Summarize decisions failed: {e}")

        try:
            recent50 = self.store.recent_decisions(50)
            self.logger.info(f"Mode stats (50): {self._mode_stats_from_decisions(recent50)}")
            self._maybe_send_periodic_summary(snapshot, recent50)
        except Exception as e:
            self.logger.warning(f"Mode stats failed: {e}")

        self._maybe_send_scorecard_summary()

        try:
            if snapshot.cash_balance_dollars >= float(self.settings.WITHDRAWAL_ALERT_THRESHOLD_DOLLARS):
                self.notifier.send(
                    f"💸 Cash ≥ ${self.settings.WITHDRAWAL_ALERT_THRESHOLD_DOLLARS:.0f}. "
                    "Consider withdrawal via Kalshi UI."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Market universe (sports-filtered)
    # ------------------------------------------------------------------
    def _fetch_market_universe(self, pages: int = 5, page_limit: int = 100) -> list[dict]:
        cursor: Optional[str] = None
        seen_tickers: set[str] = set()
        candidates: list[dict] = []
        mve_fallback: list[dict] = []
        total_seen = dropped_ns = dropped_mve = dropped_prov = dropped_q = dupes = 0

        for _ in range(max(1, int(pages))):
            try:
                raw = self.client.get_markets(limit=int(page_limit), cursor=cursor)
            except TypeError:
                raw = self.client.get_markets(limit=int(page_limit))
            except Exception as e:
                self.logger.warning(f"get_markets failed: {e}")
                break

            page_markets = extract_markets(raw)
            if not page_markets:
                break

            for m in page_markets:
                ticker = str(m.get("ticker", "") or "").strip()
                if not ticker:
                    continue
                if ticker in seen_tickers:
                    dupes += 1
                    continue
                seen_tickers.add(ticker)
                total_seen += 1

                if self.sports_only and not self._is_sports_market(m):
                    dropped_ns += 1
                    continue

                is_mve         = ticker.startswith("KXMVESPORTSMULTIGAMEEXTENDED") or bool(m.get("mve_collection_ticker"))
                is_provisional = bool(m.get("is_provisional"))

                try:
                    s = summarize_market(m)
                except Exception:
                    continue

                status_ok = str(s.get("status", "")).lower() in {"active", "open", "trading", "unknown"}
                spread = float(s.get("spread_cents", 0) or 0)
                mid    = float(s.get("mid_yes_cents", 0) or 0)
                vol    = float(s.get("volume", 0) or 0)
                quote_ok = (
                    status_ok and spread > 0
                    and spread <= int(self.settings.MAX_SPREAD_CENTS)
                    and 2 < mid < 98
                    and vol >= float(self.settings.MIN_RECENT_VOLUME)
                )

                if is_mve:
                    dropped_mve += 1
                    if quote_ok:
                        mve_fallback.append(m)
                    continue
                if is_provisional:
                    dropped_prov += 1
                    continue
                if quote_ok:
                    candidates.append(m)
                else:
                    dropped_q += 1

            if not isinstance(raw, dict):
                break
            cursor = raw.get("cursor")
            if not cursor or len(candidates) >= 150:
                break

        self.logger.info(
            "Universe | seen=%s ns_dropped=%s kept=%s mve=%s prov=%s bad_q=%s dupes=%s mve_fb=%s",
            total_seen, dropped_ns, len(candidates), dropped_mve, dropped_prov, dropped_q, dupes, len(mve_fallback),
        )

        if candidates:
            return candidates

        if mve_fallback:
            ranked = []
            for m in mve_fallback:
                try:
                    ranked.append(summarize_market(m))
                except Exception:
                    continue
            ranked.sort(key=lambda x: (
                x.get("spread_cents", 999),
                abs((x.get("mid_yes_cents", 50) or 50) - 50),
                -(x.get("volume", 0) or 0),
            ))
            self.logger.warning("MVE fallback universe (%s markets)", len(ranked))
            return [r.get("raw", {}) for r in ranked if r.get("raw")][:150]

        return []

    def _build_top_candidates(self, markets: list[dict]) -> list[dict]:
        """
        Sort the full universe by quality (low spread, mid near 50, high volume)
        and return the top N for signal evaluation.
        """
        ok_status = {"active", "open", "trading", "unknown"}
        scored: list[tuple[float, dict]] = []

        for m in markets[:300]:
            try:
                s = summarize_market(m)
            except Exception:
                continue
            if str(s.get("status", "")).lower() not in ok_status:
                continue
            spread = float(s.get("spread_cents", 0) or 0)
            mid    = float(s.get("mid_yes_cents", 0) or 0)
            vol    = float(s.get("volume", 0) or 0)

            if spread <= 0 or spread > float(self.settings.MAX_SPREAD_CENTS) * 1.5:
                continue
            if mid <= 2 or mid >= 98:
                continue

            # Lower score = better quality
            score = spread * 0.5 + abs(mid - 50.0) * 0.3 - vol * 0.0001
            s["_quality_score"] = score
            scored.append((score, s))

        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored[:TOP_CANDIDATES_TO_EVALUATE]]

    # ------------------------------------------------------------------
    # Combined exit execution (profit/stoploss + thesis deterioration)
    # ------------------------------------------------------------------
    def _maybe_execute_exit(
        self, *, now: str, chosen: dict, sig: Any, social: Any,
        news: dict, news_applied: dict, final_conviction: float, ev: dict,
    ) -> bool:
        pos = self._find_open_position_for_ticker(str(chosen.get("ticker")))
        if not pos or pos.get("_exit_pending"):
            return False

        do_pnl, pnl_reason, pnl_reasons = self._should_exit_profit_or_stoploss(
            chosen=chosen, pos=pos, final_conviction=final_conviction, ev=ev,
        )
        do_thesis, thesis_reasons = False, []
        if not do_pnl:
            do_thesis, thesis_reasons = self._should_exit_on_thesis_deterioration(
                chosen=chosen, sig=sig, final_conviction=final_conviction,
                news=news, news_applied=news_applied,
            )

        if not do_pnl and not do_thesis:
            return False

        exit_order = self._build_exit_order(chosen, pos)
        if not exit_order:
            return False

        entry_id = (pos.get("entry_id") if pos.get("_synthetic_from_decisions")
                    else self._latest_unresolved_entry_id_for_ticker(str(chosen.get("ticker"))))
        exit_trigger = pnl_reason if do_pnl else "thesis_deterioration"
        exit_action  = f"TRADE_EXIT_{exit_order['held_side']}"

        reasons = (
            list(getattr(sig, "reasons", []))
            + [f"social:{','.join(getattr(social, 'reasons', []))}",
               f"risk:exit_{exit_trigger}",
               f"news:{news.get('regime', 'unavailable')}",
               f"news_score_raw:{float(news.get('score', 0.0)):.2f}",
               f"news_score_eff:{float(news_applied.get('effective_news_score', 0.0)):.2f}",
               f"news_conf:{float(news.get('confidence', 0.0)):.2f}",
               f"exit_price_cents:{exit_order['price_cents']}",
               f"exit_order_side:{exit_order['side']}",
               f"exit_yes_no:{exit_order['yes_no']}",
               f"exit_count:{exit_order['count']}",
               f"exit_trigger:{exit_trigger}",
               f"ev_p_model:{ev.get('p_model', 0):.4f}",
               f"ev_p_mkt:{ev.get('p_market', 0):.4f}",
               f"ev_edge:{ev.get('edge', 0):.4f}",
               f"ev_cost_buf:{ev.get('cost_buffer', 0):.4f}",
               f"ev_pass:{1 if ev.get('ev_pass') else 0}",
               f"exec_mode:{ev.get('exec_mode', 'maker')}",
               ]
            + pnl_reasons + thesis_reasons
        )
        if entry_id is not None:
            reasons.append(f"entry_id:{entry_id}")

        action = "SKIP"
        try:
            self.client.place_order(
                ticker=exit_order["ticker"], side=exit_order["side"],
                yes_no=exit_order["yes_no"], count=exit_order["count"],
                price_cents=exit_order["price_cents"],
                client_order_id=exit_order["client_order_id"],
            )
            action = exit_action
            self.logger.info(
                f"EXIT {exit_action} {chosen.get('ticker')} count={exit_order['count']} "
                f"px={exit_order['price_cents']}c trigger={exit_trigger} "
                f"edge={ev.get('edge', 0):.4f} dry_run={self.settings.DRY_RUN}"
            )
            self.notifier.send(
                f"📉 {exit_action} `{chosen.get('ticker')}` "
                f"| count={exit_order['count']} px={exit_order['price_cents']}c "
                f"| trigger={exit_trigger} edge={ev.get('edge', 0):.4f} "
                f"| conv={final_conviction:.1f} dry_run={self.settings.DRY_RUN}"
            )
        except KalshiAPIError as e:
            action = "ERROR"
            self.logger.exception(f"Exit order error: {e}")
            self.notifier.send(f"⚠️ Kalshi exit order error: {e}")
        except Exception as e:
            action = "ERROR"
            self.logger.warning(f"Exit order unexpected error: {e}")

        self._record_decision_row({
            "ts": now, "ticker": chosen.get("ticker"),
            "direction": str(getattr(sig, "direction", "SKIP")),
            "base_conviction": float(getattr(sig, "conviction_score", 0.0)),
            "social_bonus": float(getattr(social, "bonus_score", 0.0)),
            "final_conviction": float(final_conviction),
            "stake_dollars": 0.0, "action": action,
            "reasons": ";".join(reasons), "dry_run": self.settings.DRY_RUN,
            "realized_pnl": None, "won": None, "resolved_ts": None,
            "market_category": chosen.get("category"),
            "news_score": float(news.get("score", 0.0)),
            "news_confidence": float(news.get("confidence", 0.0)),
            "news_regime": str(news.get("regime", "unavailable")),
            "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
            "spread_cents": chosen.get("spread_cents"),
        })
        return True

    # ------------------------------------------------------------------
    # Main tick — profit-seeking entry point
    # ------------------------------------------------------------------
    def tick(self) -> None:
        self._tick_count += 1
        now = dt.datetime.now(dt.UTC).isoformat()

        # 1. Fetch sports market universe
        try:
            markets = self._fetch_market_universe(pages=5, page_limit=100)
            self.logger.info(f"Universe: {len(markets)} sports markets")
        except Exception as e:
            self.logger.warning(f"Universe fetch failed: {e}")
            markets = []

        if not markets:
            try:
                raw     = self.client.get_markets(limit=100)
                markets = [m for m in extract_markets(raw)
                           if not self.sports_only or self._is_sports_market(m)]
                self.logger.info(f"Fallback universe: {len(markets)} markets")
            except Exception as e:
                self.logger.warning(f"Fallback fetch failed: {e}")

        snapshot = self._fetch_portfolio_snapshot()

        if not markets:
            self.logger.info("No markets available this tick.")
            try:
                recent = self.store.recent_decisions(80)
            except Exception:
                recent = []
            self._maybe_send_periodic_summary(snapshot, recent)
            self._maybe_send_scorecard_summary()
            return

        # 2. Build top N candidates
        top_candidates = self._build_top_candidates(markets)
        if not top_candidates:
            self.logger.info("No quality candidates found.")
            self._post_tick_housekeeping(snapshot)
            return

        # 3. Learning context
        try:
            recent_for_learning = self.store.recent_decisions(80)
        except Exception:
            recent_for_learning = []
        learning = self._compute_learning(recent_for_learning)

        adaptive_min_conviction = max(
            42.0,
            min(75.0, float(self.settings.MIN_CONVICTION_SCORE) + float(learning["conviction_adjustment"])),
        )
        recent_win_rate, winrate_sample = self._compute_recent_winrate_for_risk_gate(
            recent_for_learning, use_dry_run_mode=bool(self.settings.DRY_RUN), lookback_resolved=12,
        )

        # 4. Evaluate each candidate: signal → social → news → conviction
        evaluated: list[dict] = []
        final_convictions: dict[str, float] = {}

        for cand in top_candidates:
            ticker = str(cand.get("ticker", ""))
            if not ticker:
                continue

            try:
                sig = evaluate_signal(
                    cand, min_conviction=int(adaptive_min_conviction),
                    max_spread_cents=self.settings.MAX_SPREAD_CENTS,
                    min_volume=self.settings.MIN_RECENT_VOLUME,
                )
            except Exception as e:
                self.logger.warning(f"evaluate_signal failed for {ticker}: {e}")
                class _Sig:
                    direction = "SKIP"; conviction_score = 0.0; reasons = ["signal_error"]
                sig = _Sig()

            try:
                social = evaluate_social_signal(
                    cand, enabled=self.settings.SOCIAL_SIGNAL_ENABLED,
                    max_bonus=self.settings.SOCIAL_MAX_BONUS_SCORE,
                )
            except Exception as e:
                self.logger.warning(f"evaluate_social_signal failed for {ticker}: {e}")
                class _Social:
                    bonus_score = 0.0; reasons = ["social_error"]
                social = _Social()

            news = self._compute_news_signal(cand)
            final_conviction, news_applied = self._apply_news_to_conviction(
                base_signal_direction=getattr(sig, "direction", "SKIP"),
                base_conviction=float(getattr(sig, "conviction_score", 0.0)),
                social_bonus=float(getattr(social, "bonus_score", 0.0)),
                news=news,
            )
            final_convictions[ticker] = final_conviction

            # Tag candidate with evaluated state for selection below
            cand["_signal_direction"] = getattr(sig, "direction", "SKIP")
            cand["_sig_obj"]           = sig
            cand["_social_obj"]        = social
            cand["_news"]              = news
            cand["_news_applied"]      = news_applied
            cand["_final_conviction"]  = final_conviction
            evaluated.append(cand)

            self.logger.info(
                f"Eval {ticker} | dir={getattr(sig, 'direction', 'SKIP')} "
                f"conv={final_conviction:.1f} news={news['regime']} "
                f"spread={cand.get('spread_cents')} vol={cand.get('volume')}"
            )

        # 5. Exit check for every candidate where we hold a position
        open_tickers = self._all_open_position_tickers()
        for cand in evaluated:
            ticker = str(cand.get("ticker", ""))
            if ticker not in open_tickers:
                continue
            sig          = cand["_sig_obj"]
            social       = cand["_social_obj"]
            news         = cand["_news"]
            news_applied = cand["_news_applied"]
            conv         = cand["_final_conviction"]
            ev_dir       = str(cand.get("_signal_direction", "YES") or "YES").upper()
            if ev_dir == "SKIP":
                ev_dir = "YES"
            maker_price, exec_mode = self._compute_maker_entry_price(ev_dir, cand)
            ev_current = self._compute_ev_gate(ev_dir, conv, cand, maker_price, exec_mode)

            try:
                exited = self._maybe_execute_exit(
                    now=now, chosen=cand, sig=sig, social=social,
                    news=news, news_applied=news_applied,
                    final_conviction=conv, ev=ev_current,
                )
                if exited:
                    open_tickers.discard(ticker)
                    snapshot = self._fetch_portfolio_snapshot()
            except Exception as e:
                self.logger.warning(f"Exit check failed for {ticker}: {e}")

        # 6. Find the highest-EV entry not already held
        open_tickers = self._all_open_position_tickers()
        entry_candidates = [c for c in evaluated if str(c.get("ticker", "")) not in open_tickers]
        best_ev_result = self._select_best_ev_candidate(entry_candidates, final_convictions)

        if best_ev_result is None:
            # Log skip rows for all evaluated candidates
            for cand in evaluated:
                ticker = str(cand.get("ticker", ""))
                sig_dir = str(cand.get("_signal_direction", "SKIP") or "SKIP").upper()
                conv    = float(cand.get("_final_conviction", 0.0))
                ev_dir  = sig_dir if sig_dir != "SKIP" else "YES"
                maker_p, e_mode = self._compute_maker_entry_price(ev_dir, cand)
                ev_info = self._compute_ev_gate(ev_dir, conv, cand, maker_p, e_mode)
                self.logger.info(
                    f"SKIP {ticker} | dir={sig_dir} conv={conv:.1f} "
                    f"edge={ev_info['edge']:.4f} tier={ev_info['tier']}"
                )
                self._record_decision_row({
                    "ts": now, "ticker": ticker, "direction": sig_dir,
                    "base_conviction": float(getattr(cand.get("_sig_obj"), "conviction_score", 0.0)),
                    "social_bonus": float(getattr(cand.get("_social_obj"), "bonus_score", 0.0)),
                    "final_conviction": conv, "stake_dollars": 0.0, "action": "SKIP",
                    "reasons": ";".join([
                        f"skip:no_ev_pass",
                        f"ev_edge:{ev_info['edge']:.4f}",
                        f"ev_tier:{ev_info['tier']}",
                        f"ev_p_model:{ev_info['p_model']:.4f}",
                        f"ev_p_mkt:{ev_info['p_market']:.4f}",
                        f"ev_cost_buf:{ev_info['cost_buffer']:.4f}",
                        f"exec_mode:{e_mode}",
                    ]),
                    "dry_run": self.settings.DRY_RUN,
                    "realized_pnl": None, "won": None, "resolved_ts": None,
                    "market_category": cand.get("category"),
                    "news_score": float(cand["_news"].get("score", 0.0)),
                    "news_confidence": float(cand["_news"].get("confidence", 0.0)),
                    "news_regime": str(cand["_news"].get("regime", "unavailable")),
                    "news_effective_score": float(cand["_news_applied"].get("effective_news_score", 0.0)),
                    "spread_cents": cand.get("spread_cents"),
                })
            self._post_tick_housekeeping(snapshot)
            return

        chosen, sig_dir, limit_price, exec_mode, ev = best_ev_result
        ticker       = str(chosen.get("ticker", ""))
        sig          = chosen["_sig_obj"]
        social       = chosen["_social_obj"]
        news         = chosen["_news"]
        news_applied = chosen["_news_applied"]
        final_conviction = chosen["_final_conviction"]
        market_label = str(
            chosen.get("market_label") or chosen.get("title")
            or chosen.get("question") or chosen.get("name") or ticker
        ).strip()

        self.logger.info(
            f"Best EV: {ticker} dir={sig_dir} conv={final_conviction:.1f} "
            f"edge={ev['edge']:.4f} tier={ev['tier']} exec={exec_mode} px={limit_price}c"
        )

        # 7. Entry guards
        skip_reason = ""
        if self._daily_loss_cap_breached():
            skip_reason = "daily_loss_cap_breached"
        elif len(self._all_open_position_tickers()) >= MAX_SIMULTANEOUS_POSITIONS:
            skip_reason = "max_positions_reached"
        elif self._find_open_position_for_ticker(ticker) is not None:
            skip_reason = "already_have_open_position"
        elif self._is_in_cooldown(ticker):
            skip_reason = f"cooldown_not_elapsed({ENTRY_COOLDOWN_MINUTES}m)"
        elif snapshot.cash_balance_dollars < float(self.settings.MIN_TRADE_DOLLARS):
            skip_reason = f"bankroll_too_low({snapshot.cash_balance_dollars:.2f})"

        if skip_reason:
            self.logger.info(f"SKIP {ticker} | guard: {skip_reason}")
            self._record_decision_row({
                "ts": now, "ticker": ticker, "direction": sig_dir,
                "base_conviction": float(getattr(sig, "conviction_score", 0.0)),
                "social_bonus": float(getattr(social, "bonus_score", 0.0)),
                "final_conviction": float(final_conviction),
                "stake_dollars": 0.0, "action": "SKIP",
                "reasons": (
                    f"skip:guard_{skip_reason};"
                    f"ev_edge:{ev['edge']:.4f};ev_tier:{ev['tier']};ev_pass:1;exec_mode:{exec_mode}"
                ),
                "dry_run": self.settings.DRY_RUN,
                "realized_pnl": None, "won": None, "resolved_ts": None,
                "market_category": chosen.get("category"),
                "news_score": float(news.get("score", 0.0)),
                "news_confidence": float(news.get("confidence", 0.0)),
                "news_regime": str(news.get("regime", "unavailable")),
                "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
                "spread_cents": chosen.get("spread_cents"),
            })
            self._post_tick_housekeeping(snapshot)
            return

        # 8. Risk engine (final safety check + stake base)
        social_size_boost = 0.0
        try:
            sb = float(getattr(social, "bonus_score", 0.0))
            if sb > 0:
                social_size_boost = min(float(self.settings.SOCIAL_SIZE_BOOST_MAX_PCT), sb / 100.0)
        except Exception:
            pass

        try:
            risk = evaluate_risk(
                conviction=final_conviction,
                current_open_exposure=snapshot.open_exposure_dollars,
                open_positions_count=snapshot.open_positions_count,
                daily_pnl=snapshot.daily_pnl_dollars,
                available_cash_dollars=snapshot.cash_balance_dollars,
                min_dollars=self.settings.MIN_TRADE_DOLLARS,
                max_dollars=self.settings.MAX_TRADE_DOLLARS,
                daily_max_loss_dollars=self.settings.DAILY_MAX_LOSS_DOLLARS,
                max_open_exposure_dollars=self.settings.MAX_OPEN_EXPOSURE_DOLLARS,
                max_simultaneous_positions=MAX_SIMULTANEOUS_POSITIONS,
                social_size_boost_pct=social_size_boost,
                learning_multiplier=float(learning["stake_multiplier"]),
                recent_win_rate=recent_win_rate,
                min_win_rate_for_risk_on=self.min_winrate_for_risk_on_gate,
            )
        except Exception as e:
            self.logger.warning(f"evaluate_risk failed: {e}")
            class _Risk:
                allowed = False; reason = "risk_error_fallback"; stake_dollars = 0.0
            risk = _Risk()

        if not getattr(risk, "allowed", False):
            reason_str = getattr(risk, "reason", "unknown")
            self.logger.info(f"SKIP {ticker} | risk denied: {reason_str}")
            self._record_decision_row({
                "ts": now, "ticker": ticker, "direction": sig_dir,
                "base_conviction": float(getattr(sig, "conviction_score", 0.0)),
                "social_bonus": float(getattr(social, "bonus_score", 0.0)),
                "final_conviction": float(final_conviction),
                "stake_dollars": 0.0, "action": "SKIP",
                "reasons": f"skip:risk_denied({reason_str});ev_edge:{ev['edge']:.4f};ev_pass:1",
                "dry_run": self.settings.DRY_RUN,
                "realized_pnl": None, "won": None, "resolved_ts": None,
                "market_category": chosen.get("category"),
                "news_score": float(news.get("score", 0.0)),
                "news_confidence": float(news.get("confidence", 0.0)),
                "news_regime": str(news.get("regime", "unavailable")),
                "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
                "spread_cents": chosen.get("spread_cents"),
            })
            self._post_tick_housekeeping(snapshot)
            return

        # 9. Expert Kelly sizing — take the MORE aggressive of risk engine vs Kelly
        kelly_stake, kelly_contracts = self._compute_expert_kelly_stake(
            ev=ev, bankroll_dollars=snapshot.cash_balance_dollars,
        )
        risk_stake = float(getattr(risk, "stake_dollars", MIN_STAKE_DOLLARS))

        # Be greedy: use the larger number, then cap it
        tier = ev.get("tier", "low")
        if tier == "elite":
            max_cap, max_ct = MAX_STAKE_DOLLARS_ELITE,  MAX_CONTRACTS_ELITE
        elif tier == "high":
            max_cap, max_ct = MAX_STAKE_DOLLARS_STRONG, MAX_CONTRACTS_STRONG
        else:
            max_cap, max_ct = MAX_STAKE_DOLLARS_NORMAL, MAX_CONTRACTS_NORMAL

        raw_stake     = max(risk_stake, kelly_stake)
        stake_dollars = min(max_cap, max(MIN_STAKE_DOLLARS, raw_stake))
        stake_dollars = min(stake_dollars, snapshot.cash_balance_dollars * BANKROLL_PCT_CAP)

        contracts = max(1, min(max_ct, int(round((stake_dollars * 100.0) / max(1, limit_price)))))
        stake_dollars = round((contracts * limit_price) / 100.0, 2)

        # 10. Build and place the order
        order = {
            "ticker": ticker,
            "side": "buy",
            "yes_no": "yes" if sig_dir == "YES" else "no",
            "count": contracts,
            "price_cents": int(max(1, min(99, limit_price))),
            "client_order_id": f"entry-{ticker}-{int(dt.datetime.now(dt.UTC).timestamp())}",
        }

        reasons: list[str] = (
            list(getattr(sig, "reasons", []))
            + [
                f"social:{','.join(getattr(social, 'reasons', []))}",
                f"risk:{getattr(risk, 'reason', 'ok')}",
                f"learning_mode:{learning['mode']}",
                f"news:{news['regime']}",
                f"news_score_raw:{float(news.get('score', 0.0)):.2f}",
                f"news_score_eff:{float(news_applied.get('effective_news_score', 0.0)):.2f}",
                f"news_conf:{float(news.get('confidence', 0.0)):.2f}",
                f"news_conf_scale:{float(news_applied.get('news_conf_scale', 0.0)):.2f}",
                f"news_dir_scale:{float(news_applied.get('news_direction_scale', 0.0)):.2f}",
                f"news_conflict_sig:{1 if news_applied.get('news_conflicts_signal') else 0}",
                f"ev_p_model:{ev['p_model']:.4f}",
                f"ev_p_mkt:{ev['p_market']:.4f}",
                f"ev_edge:{ev['edge']:.4f}",
                f"ev_cost_buf:{ev['cost_buffer']:.4f}",
                f"ev_pass:1",
                f"ev_tier:{tier}",
                f"exec_mode:{exec_mode}",
                f"entry_price_cents:{order['price_cents']}",
                f"order_count:{contracts}",
                f"order_side:{order['side']}",
                f"order_yes_no:{order['yes_no']}",
                f"kelly_stake:{kelly_stake:.2f}",
                f"kelly_contracts:{kelly_contracts}",
                f"final_stake:{stake_dollars:.2f}",
                f"mid_yes_cents:{float(chosen.get('mid_yes_cents', 0) or 0):.2f}",
                f"spread_cents:{ev['spread_cents']:.1f}",
                f"volume:{ev['volume']:.0f}",
            ]
        )
        if recent_win_rate is not None:
            reasons.append(f"risk_gate_wr:{recent_win_rate:.3f};risk_gate_wr_n:{winrate_sample}")
        if market_label:
            reasons.append(f"market_label:{market_label[:140]}")
        if news.get("risk_flags"):
            reasons.append(f"news_flags:{','.join(news['risk_flags'])}")

        action = "SKIP"
        try:
            self.client.place_order(
                ticker=order["ticker"], side=order["side"],
                yes_no=order["yes_no"], count=order["count"],
                price_cents=order["price_cents"],
                client_order_id=order["client_order_id"],
            )
            action = f"TRADE_{sig_dir}"
            self._set_entry_cooldown(ticker)
            self.logger.info(
                f"{action} {ticker} stake=${stake_dollars:.2f} px={limit_price}c "
                f"x{contracts} edge={ev['edge']:.4f} tier={tier} exec={exec_mode} "
                f"dry_run={self.settings.DRY_RUN}"
            )
            self.notifier.send(
                f"📈 {action} `{ticker}` | ${stake_dollars:.2f} "
                f"| conv={final_conviction:.1f} "
                f"(base={float(getattr(sig, 'conviction_score', 0.0)):.1f} "
                f"soc={float(getattr(social, 'bonus_score', 0.0)):.1f} "
                f"news={news_applied.get('effective_news_score', 0.0):.1f}) "
                f"| edge={ev['edge']:.4f} tier={tier} exec={exec_mode} "
                f"| px={limit_price}c x{contracts} "
                f"| kelly×{KELLY_TIER_ELITE:.0%} cap "
                f"| news={news['regime']} "
                f"| wr={'n/a' if recent_win_rate is None else f'{recent_win_rate*100:.1f}%'} "
                f"| dry_run={self.settings.DRY_RUN}"
            )
        except KalshiAPIError as e:
            action = "ERROR"
            stake_dollars = 0.0
            self.logger.exception(f"Order error: {e}")
            self.notifier.send(f"⚠️ Kalshi order error: {e}")
        except Exception as e:
            action = "ERROR"
            stake_dollars = 0.0
            self.logger.warning(f"Order unexpected error: {e}")

        self._record_decision_row({
            "ts": now, "ticker": ticker, "direction": sig_dir,
            "base_conviction": float(getattr(sig, "conviction_score", 0.0)),
            "social_bonus": float(getattr(social, "bonus_score", 0.0)),
            "final_conviction": float(final_conviction),
            "stake_dollars": stake_dollars if action.startswith("TRADE_") else 0.0,
            "action": action, "reasons": ";".join(reasons),
            "dry_run": self.settings.DRY_RUN,
            "realized_pnl": None, "won": None, "resolved_ts": None,
            "market_category": chosen.get("category"),
            "news_score": float(news.get("score", 0.0)),
            "news_confidence": float(news.get("confidence", 0.0)),
            "news_regime": str(news.get("regime", "unavailable")),
            "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
            "spread_cents": chosen.get("spread_cents"),
        })

        self._post_tick_housekeeping(snapshot)

    def main(self) -> None:
        raise NotImplementedError("Use module-level main().")


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