from __future__ import annotations

import datetime as dt
import signal
import sys
from typing import Any, Optional

from app.config import Settings
from app.discord_notifier import DiscordNotifier
from app.execution_engine import build_limit_order_from_signal
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

# Optional lightweight adaptive module (safe fallback if file not added yet)
try:
    from app.learning import compute_learning_adjustment
except Exception:
    compute_learning_adjustment = None  # type: ignore

# Optional news/trend confirmation module (safe fallback)
try:
    from app.news_signal import evaluate_news_signal
except Exception:
    evaluate_news_signal = None  # type: ignore

# Optional setup scorecard summary module (safe fallback)
try:
    from app.setup_scorecard import build_setup_scorecard, summarize_scorecard_for_discord
except Exception:
    build_setup_scorecard = None  # type: ignore
    summarize_scorecard_for_discord = None  # type: ignore


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
        self._last_summary_sent_minute = None

        # Paper trade resolver settings (MVP defaults; move to Settings later if you want)
        self.paper_resolver_enabled = True
        self.paper_resolver_min_age_minutes = 15
        self.paper_resolver_max_to_check = 25

        # Risk-on suppression inputs (not trade blocking)
        self.min_resolved_for_winrate_gate = 6
        self.min_winrate_for_risk_on_gate = 0.48

        # News signal defaults (bounded, optional)
        self.news_signal_enabled = True
        self.news_max_abs_score = 8.0
        self.news_require_two_credible_sources = True

        # News application controls (important: keeps core signal dominant)
        self.news_min_confidence_to_apply = 0.35          # below this -> mostly ignore
        self.news_full_confidence_at = 0.75               # at/above this -> full effect
        self.news_conflict_penalty_multiplier = 0.35      # conflicting news only partially counts
        self.news_neutral_regime_multiplier = 0.40        # noisy/weak regime gets heavily damped
        self.news_mixed_regime_multiplier = 0.60          # mixed/conflicting reports damped
        self.news_abs_effect_cap_on_conviction = 6.0      # final effective news impact cap
        self.news_allow_direction_flip = False            # don't let news reverse trade direction logic

        # Scorecard cadence (every N ticks)
        self.scorecard_summary_every_n_ticks = 30

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

        try:
            pos = self.client.get_positions()
            if isinstance(pos, dict):
                positions = pos.get("positions") if isinstance(pos.get("positions"), list) else []
                snap.open_positions_count = len(positions)
                snap.open_exposure_dollars = min(
                    self.settings.MAX_OPEN_EXPOSURE_DOLLARS,
                    len(positions) * self.settings.MIN_TRADE_DOLLARS,
                )
        except Exception as e:
            self.logger.warning(f"Positions fetch failed: {e}")

        snap.daily_pnl_dollars = 0.0
        return snap

    def _choose_market(self, markets: list[dict]) -> Any:
        candidates = []
        for m in markets[:200]:
            s = summarize_market(m)

            if s["status"].lower() not in {"active", "open", "trading", "unknown"}:
                continue

            if s["spread_cents"] <= 0 or s["spread_cents"] > self.settings.MAX_SPREAD_CENTS:
                continue
            if s["volume"] < self.settings.MIN_RECENT_VOLUME:
                continue

            mid = s["mid_yes_cents"]
            if mid <= 2 or mid >= 98:
                continue

            candidates.append(s)

        candidates.sort(key=lambda x: (x["spread_cents"], abs(x["mid_yes_cents"] - 50), -x["volume"]))
        return candidates[0] if candidates else None

    def _compute_learning(self, recent: list[dict]) -> dict:
        if compute_learning_adjustment is None:
            return {
                "stake_multiplier": 1.0,
                "conviction_adjustment": 0.0,
                "mode": "disabled",
                "reasons": ["learning_module_missing"],
            }

        try:
            adj = compute_learning_adjustment(recent)
            return {
                "stake_multiplier": float(getattr(adj, "stake_multiplier", 1.0)),
                "conviction_adjustment": float(getattr(adj, "conviction_adjustment", 0.0)),
                "mode": str(getattr(adj, "mode", "neutral")),
                "reasons": list(getattr(adj, "reasons", [])),
            }
        except Exception as e:
            self.logger.warning(f"Learning adjustment failed, using neutral: {e}")
            return {
                "stake_multiplier": 1.0,
                "conviction_adjustment": 0.0,
                "mode": "error_fallback",
                "reasons": ["learning_error"],
            }

    def _mode_stats_from_decisions(self, decisions: list[dict]) -> dict:
        def summarize_mode(rows: list[dict]) -> dict:
            total = len(rows)
            trades = [r for r in rows if str(r.get("action", "")).startswith("TRADE")]
            skips = [r for r in rows if r.get("action") == "SKIP"]
            errors = [r for r in rows if r.get("action") == "ERROR"]

            avg_conv = round(sum(float(r.get("final_conviction", 0.0)) for r in rows) / total, 2) if total else 0.0
            avg_stake = round(sum(float(r.get("stake_dollars", 0.0)) for r in trades) / len(trades), 2) if trades else 0.0

            resolved = [r for r in trades if r.get("resolved_ts") is not None]
            pnl_values = [r.get("realized_pnl") for r in resolved if r.get("realized_pnl") is not None]
            wins = [r for r in resolved if r.get("won") is True or r.get("won") == 1]
            losses = [r for r in resolved if r.get("won") is False or r.get("won") == 0]

            pnl_total = round(sum(float(x) for x in pnl_values), 2) if pnl_values else None
            win_rate = round((len(wins) / (len(wins) + len(losses))) * 100.0, 1) if (wins or losses) else None
            expectancy = round((sum(float(x) for x in pnl_values) / len(pnl_values)), 4) if pnl_values else None

            return {
                "count": total,
                "trades": len(trades),
                "skips": len(skips),
                "errors": len(errors),
                "resolved_trades": len(resolved),
                "avg_conviction": avg_conv,
                "avg_stake": avg_stake,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate,
                "pnl_total": pnl_total,
                "expectancy": expectancy,
            }

        sim_rows = [d for d in decisions if bool(d.get("dry_run", True))]
        live_rows = [d for d in decisions if not bool(d.get("dry_run", True))]
        return {"sim": summarize_mode(sim_rows), "live": summarize_mode(live_rows)}

    def _compute_recent_winrate_for_risk_gate(
        self,
        decisions: list[dict],
        *,
        use_dry_run_mode: bool,
        lookback_resolved: int = 12,
    ) -> tuple[Optional[float], int]:
        mode_rows = [d for d in decisions if bool(d.get("dry_run", True)) == bool(use_dry_run_mode)]
        resolved_trades = [
            d for d in mode_rows
            if str(d.get("action", "")).startswith("TRADE") and d.get("resolved_ts") is not None
        ]

        wins = 0
        losses = 0
        counted = 0
        for d in resolved_trades[:lookback_resolved]:
            w = d.get("won")
            if w is True or w == 1:
                wins += 1
                counted += 1
            elif w is False or w == 0:
                losses += 1
                counted += 1

        if counted < self.min_resolved_for_winrate_gate:
            return None, counted

        return (wins / counted) if counted > 0 else None, counted

    # -----------------------------
    # News / scorecard helpers
    # -----------------------------
    def _news_category_allowlist(self) -> list[str]:
        return ["politics", "weather", "sports", "macro"]

    def _fetch_news_items_for_market(self, chosen: dict) -> list[dict]:
        """
        Placeholder adapter for future real news integration.
        """
        return []

    def _compute_news_signal(self, chosen: dict) -> dict:
        if not self.news_signal_enabled or evaluate_news_signal is None:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "regime": "disabled",
                "reasons": ["news_disabled_or_module_missing"],
                "risk_flags": [],
                "n_items": 0,
                "n_unique": 0,
                "n_tier1": 0,
                "n_tier2": 0,
                "n_tier3": 0,
            }

        try:
            news_items = self._fetch_news_items_for_market(chosen)
        except Exception as e:
            self.logger.warning(f"News fetch failed for {chosen.get('ticker')}: {e}")
            news_items = []

        try:
            ns = evaluate_news_signal(
                chosen,
                news_items,
                enabled=True,
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
            self.logger.warning(f"News signal evaluation failed; using neutral: {e}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "regime": "error_fallback",
                "reasons": ["news_eval_error"],
                "risk_flags": [],
                "n_items": 0,
                "n_unique": 0,
                "n_tier1": 0,
                "n_tier2": 0,
                "n_tier3": 0,
            }

    def _apply_news_to_conviction(
        self,
        *,
        base_signal_direction: str,
        base_conviction: float,
        social_bonus: float,
        news: dict,
    ) -> tuple[float, dict]:
        """
        Convert raw news score into an *effective* conviction adjustment.

        Design goals:
        - news should help, not dominate
        - low-confidence/noisy news should barely move anything
        - conflicting news should dampen, not hijack, by default
        """
        raw_news_score = float(news.get("score", 0.0))
        news_conf = max(0.0, min(1.0, float(news.get("confidence", 0.0))))
        news_regime = str(news.get("regime", "unavailable"))
        sig_dir = str(base_signal_direction or "SKIP").upper()

        # No directional trade -> still compute for logging, but don't force action logic
        # (it only affects final conviction if a trade signal already exists downstream)
        # Confidence scaling with floor
        if news_conf <= self.news_min_confidence_to_apply:
            conf_scale = 0.10  # tiny effect, not zero (useful for logging/experiments)
        elif news_conf >= self.news_full_confidence_at:
            conf_scale = 1.00
        else:
            span = max(1e-9, self.news_full_confidence_at - self.news_min_confidence_to_apply)
            conf_scale = 0.10 + 0.90 * ((news_conf - self.news_min_confidence_to_apply) / span)

        # Regime scaling
        regime_scale = 1.0
        if news_regime in {"noisy_or_weak", "unavailable", "disabled", "error_fallback"}:
            regime_scale = self.news_neutral_regime_multiplier
        elif news_regime in {"mixed"}:
            regime_scale = self.news_mixed_regime_multiplier

        # Directional compatibility check (raw news score >0 implies YES-ish; <0 implies NO-ish)
        # If signal is YES and news is negative (or vice versa), dampen heavily.
        conflict = False
        if sig_dir == "YES" and raw_news_score < 0:
            conflict = True
        elif sig_dir == "NO" and raw_news_score > 0:
            conflict = True

        direction_scale = self.news_conflict_penalty_multiplier if conflict else 1.0

        effective_news_score = raw_news_score * conf_scale * regime_scale * direction_scale

        # Optional safety: do not allow news to push conviction so far that it effectively flips a borderline setup
        if (not self.news_allow_direction_flip) and conflict:
            # Clamp conflicting contribution more aggressively
            effective_news_score = max(-2.0, min(2.0, effective_news_score))

        # Final cap on conviction impact
        effective_news_score = max(
            -self.news_abs_effect_cap_on_conviction,
            min(self.news_abs_effect_cap_on_conviction, effective_news_score),
        )

        final_conviction = min(
            100.0,
            max(0.0, float(base_conviction) + float(social_bonus) + effective_news_score),
        )

        meta = {
            "raw_news_score": round(raw_news_score, 3),
            "effective_news_score": round(float(effective_news_score), 3),
            "news_confidence": round(news_conf, 3),
            "news_regime": news_regime,
            "news_conf_scale": round(conf_scale, 3),
            "news_regime_scale": round(regime_scale, 3),
            "news_direction_scale": round(direction_scale, 3),
            "news_conflicts_signal": conflict,
        }
        return final_conviction, meta

    def _extract_recent_setup_scorecard(self, lookback: int = 200) -> Optional[dict]:
        if build_setup_scorecard is None:
            return None
        try:
            decisions = self.store.recent_decisions(lookback)
            return build_setup_scorecard(decisions)
        except Exception as e:
            self.logger.warning(f"Build setup scorecard failed: {e}")
            return None

    def _maybe_send_scorecard_summary(self) -> None:
        if build_setup_scorecard is None or summarize_scorecard_for_discord is None:
            return
        if self.scorecard_summary_every_n_ticks <= 0:
            return
        if (self._tick_count % self.scorecard_summary_every_n_ticks) != 0:
            return

        try:
            scorecard = self._extract_recent_setup_scorecard(250)
            if not scorecard:
                return
            msg = summarize_scorecard_for_discord(scorecard, top_n=3)
            self.notifier.send(msg)
        except Exception as e:
            self.logger.warning(f"Scorecard summary failed: {e}")

    def _maybe_send_periodic_summary(self, snapshot: PortfolioSnapshot, recent: list[dict]) -> None:
        if not recent:
            return
        if (self._tick_count % 10) != 0:
            return

        mode_stats = self._mode_stats_from_decisions(recent)
        sim = mode_stats["sim"]
        live = mode_stats["live"]

        def _fmt_mode(name: str, m: dict) -> str:
            wl = (
                f"W/L {m['wins']}-{m['losses']} ({m['win_rate']}%)"
                if m["win_rate"] is not None
                else "W/L n/a"
            )
            pnl = f"PnL ${m['pnl_total']:.2f}" if m["pnl_total"] is not None else "PnL n/a"
            exp = f"Exp ${m['expectancy']:.4f}" if m["expectancy"] is not None else "Exp n/a"
            return (
                f"{name}: trades={m['trades']} resolved={m['resolved_trades']} skips={m['skips']} errors={m['errors']} "
                f"avg_conv={m['avg_conviction']:.1f} avg_stake=${m['avg_stake']:.2f} {wl} {pnl} {exp}"
            )

        msg = (
            "📊 Bot Summary\n"
            f"Env={self.settings.KALSHI_ENV} | DRY_RUN={self.settings.DRY_RUN} | "
            f"cash≈${snapshot.cash_balance_dollars:.2f} | open_exp≈${snapshot.open_exposure_dollars:.2f}\n"
            f"{_fmt_mode('SIM', sim)}\n"
            f"{_fmt_mode('LIVE', live)}"
        )
        self.notifier.send(msg)

    def tick(self) -> None:
        self._tick_count += 1
        now = dt.datetime.now(dt.UTC).isoformat()

        raw = self.client.get_markets(limit=100)
        markets = extract_markets(raw)
        self.logger.info(f"Fetched markets: {len(markets)}")
        if not markets:
            return

        chosen = self._choose_market(markets)
        if not chosen:
            self.logger.info("No candidate market found.")
            snapshot = self._fetch_portfolio_snapshot()
            recent = self.store.recent_decisions(80)
            self._maybe_send_periodic_summary(snapshot, recent)
            self._maybe_send_scorecard_summary()
            return

        self.logger.info(
            f"Candidate {chosen['ticker']} | mid={chosen['mid_yes_cents']:.1f} "
            f"| spread={chosen['spread_cents']} | vol={chosen['volume']}"
        )

        recent_for_learning = self.store.recent_decisions(80)
        learning = self._compute_learning(recent_for_learning)

        self.logger.info(
            f"Learning mode={learning['mode']} | stake_mult={learning['stake_multiplier']:.2f} "
            f"| conv_adj={learning['conviction_adjustment']:.1f} | reasons={learning['reasons']}"
        )

        adaptive_min_conviction = max(
            45,
            min(80, float(self.settings.MIN_CONVICTION_SCORE) + float(learning["conviction_adjustment"])),
        )

        sig = evaluate_signal(
            chosen,
            min_conviction=int(adaptive_min_conviction),
            max_spread_cents=self.settings.MAX_SPREAD_CENTS,
            min_volume=self.settings.MIN_RECENT_VOLUME,
        )

        social = evaluate_social_signal(
            chosen,
            enabled=self.settings.SOCIAL_SIGNAL_ENABLED,
            max_bonus=self.settings.SOCIAL_MAX_BONUS_SCORE,
        )

        news = self._compute_news_signal(chosen)
        self.logger.info(
            f"News signal regime={news['regime']} raw_score={news['score']:.2f} conf={news['confidence']:.2f} "
            f"items={news['n_items']}/{news['n_unique']} t1={news['n_tier1']} t2={news['n_tier2']} t3={news['n_tier3']} "
            f"reasons={news['reasons']} flags={news['risk_flags']}"
        )

        # Apply news safely (confidence + regime + direction aware)
        final_conviction, news_applied = self._apply_news_to_conviction(
            base_signal_direction=sig.direction,
            base_conviction=float(sig.conviction_score),
            social_bonus=float(social.bonus_score),
            news=news,
        )

        self.logger.info(
            "News applied "
            f"raw={news_applied['raw_news_score']:.2f} eff={news_applied['effective_news_score']:.2f} "
            f"conf_scale={news_applied['news_conf_scale']:.2f} regime_scale={news_applied['news_regime_scale']:.2f} "
            f"dir_scale={news_applied['news_direction_scale']:.2f} conflict={news_applied['news_conflicts_signal']}"
        )

        snapshot = self._fetch_portfolio_snapshot()

        social_size_boost = 0.0
        if sig.direction != "SKIP" and social.bonus_score > 0:
            social_size_boost = min(self.settings.SOCIAL_SIZE_BOOST_MAX_PCT, social.bonus_score / 100.0)

        recent_win_rate_for_gate, winrate_sample = self._compute_recent_winrate_for_risk_gate(
            recent_for_learning,
            use_dry_run_mode=self.settings.DRY_RUN,
            lookback_resolved=12,
        )

        if recent_win_rate_for_gate is None:
            self.logger.info(
                f"Risk-on gate: inactive (resolved sample too small: {winrate_sample}/{self.min_resolved_for_winrate_gate})"
            )
        else:
            self.logger.info(
                f"Risk-on gate: recent_wr={recent_win_rate_for_gate:.2%} sample={winrate_sample} "
                f"threshold={self.min_winrate_for_risk_on_gate:.2%}"
            )

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
            learning_multiplier=float(learning["stake_multiplier"]),
            recent_win_rate=recent_win_rate_for_gate,
            min_win_rate_for_risk_on=self.min_winrate_for_risk_on_gate,
        )

        action = "SKIP"
        order_resp = None
        reasons = sig.reasons + [
            f"social:{','.join(social.reasons)}",
            f"risk:{risk.reason}",
            f"learning_mode:{learning['mode']}",
            f"news:{news['regime']}",
            f"news_score_raw:{float(news.get('score', 0.0)):.2f}",
            f"news_score_eff:{float(news_applied.get('effective_news_score', 0.0)):.2f}",
            f"news_conf:{float(news.get('confidence', 0.0)):.2f}",
            f"news_conf_scale:{float(news_applied.get('news_conf_scale', 0.0)):.2f}",
            f"news_regime_scale:{float(news_applied.get('news_regime_scale', 0.0)):.2f}",
            f"news_dir_scale:{float(news_applied.get('news_direction_scale', 0.0)):.2f}",
            f"news_conflict_sig:{1 if news_applied.get('news_conflicts_signal') else 0}",
        ]

        if news.get("risk_flags"):
            reasons.append(f"news_flags:{','.join(news['risk_flags'])}")

        if recent_win_rate_for_gate is not None:
            reasons.append(f"risk_gate_wr:{recent_win_rate_for_gate:.3f}")
            reasons.append(f"risk_gate_wr_n:{winrate_sample}")

        if sig.direction != "SKIP" and risk.allowed:
            order = build_limit_order_from_signal(chosen["ticker"], sig.direction, chosen, risk.stake_dollars)

            try:
                reasons.append(f"entry_price_cents:{int(order['price_cents'])}")
            except Exception:
                pass

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
                    f"| conv={final_conviction:.1f} "
                    f"(base {sig.conviction_score:.1f} + social {social.bonus_score:.1f} + news_eff {float(news_applied['effective_news_score']):.1f}) "
                    f"| learning={learning['mode']} x{float(learning['stake_multiplier']):.2f} "
                    f"| news={news['regime']} conf={float(news.get('confidence', 0.0)):.2f} "
                    f"| gate_wr={('n/a' if recent_win_rate_for_gate is None else f'{recent_win_rate_for_gate*100:.1f}%')} "
                    f"| dry_run={self.settings.DRY_RUN}"
                )

            except KalshiAPIError as e:
                action = "ERROR"
                self.logger.exception(f"Order error: {e}")
                self.notifier.send(f"⚠️ Kalshi order error: {e}")
        else:
            self.logger.info(
                f"SKIP {chosen['ticker']} | dir={sig.direction} | conv={final_conviction:.1f} | risk={risk.reason}"
            )

        row = {
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
            "realized_pnl": None,
            "won": None,
            "resolved_ts": None,
            "market_category": chosen.get("category"),
            # Optional newer columns (safe if StateStore supports them)
            "news_score": float(news.get("score", 0.0)),
            "news_confidence": float(news.get("confidence", 0.0)),
            "news_regime": str(news.get("regime", "unavailable")),
            "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
            "spread_cents": chosen.get("spread_cents"),
        }

        try:
            _inserted_id = self.store.insert_decision(row)
        except TypeError:
            # Backward compatibility if older StateStore insert signature ignores new fields poorly
            legacy_row = {
                "ts": row["ts"],
                "ticker": row["ticker"],
                "direction": row["direction"],
                "base_conviction": row["base_conviction"],
                "social_bonus": row["social_bonus"],
                "final_conviction": row["final_conviction"],
                "stake_dollars": row["stake_dollars"],
                "action": row["action"],
                "reasons": row["reasons"],
                "dry_run": row["dry_run"],
                "realized_pnl": row["realized_pnl"],
                "won": row["won"],
                "resolved_ts": row["resolved_ts"],
                "market_category": row["market_category"],
            }
            _inserted_id = self.store.insert_decision(legacy_row)

        if self.settings.DRY_RUN and self.paper_resolver_enabled:
            try:
                rr = resolve_paper_trades(
                    store=self.store,
                    client=self.client,
                    logger=self.logger,
                    max_to_check=self.paper_resolver_max_to_check,
                    min_age_minutes=self.paper_resolver_min_age_minutes,
                )
                if rr.updated > 0:
                    self.logger.info(
                        f"Paper resolver updated={rr.updated} checked={rr.checked} skipped={rr.skipped}"
                    )
            except Exception as e:
                self.logger.warning(f"Paper resolver failed: {e}")

        recent20 = self.store.recent_decisions(20)
        summary = summarize_decisions(recent20)
        self.logger.info(f"Recent summary: {summary}")

        recent50 = self.store.recent_decisions(50)
        mode_stats = self._mode_stats_from_decisions(recent50)
        self.logger.info(f"Mode stats (50): {mode_stats}")

        self._maybe_send_periodic_summary(snapshot, recent50)
        self._maybe_send_scorecard_summary()

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
