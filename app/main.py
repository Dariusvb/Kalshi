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

        # Conservative exit-management (thesis deterioration only; avoid paper hands)
        self.exit_management_enabled = True
        self.exit_min_hold_minutes = 5                    # avoid immediate churn after entry
        self.exit_max_hold_conviction = 52.0              # if thesis drops to weak/skip-ish
        self.exit_conviction_drop_points = 12.0           # large deterioration from entry conviction
        self.exit_require_opposite_or_skip = True         # won't exit just because conviction wiggles
        self.exit_news_conflict_min_conf = 0.75           # only trust strong/confident conflicting news
        self.exit_news_conflict_min_effect = 2.5          # adverse effective news score threshold

        # Scorecard cadence (every N ticks)
        self.scorecard_summary_every_n_ticks = 30

        # DRY_RUN-only pipeline validation knobs (safe defaults; live untouched)
        self.pipeline_test_entries_enabled = True
        self.pipeline_test_max_entries = 3
        self.pipeline_test_min_ticks_between = 3
        self._pipeline_test_entries_sent = 0
        self._pipeline_test_last_entry_tick: Optional[int] = None

        # -----------------------------
        # Patch 2: NewsCache wiring (uses your app/news_cache.py class)
        # -----------------------------
        self.news_cache = None
        try:
            import os
            from app.news_cache import NewsCache

            # Store cache next to the DB file, so it persists on the same volume.
            cache_path = os.path.join(os.path.dirname(self.settings.DB_PATH), "news_cache.json")
            self.news_cache = NewsCache(path=cache_path, ttl_seconds=6 * 3600)
        except Exception:
            self.news_cache = None

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
            positions = self._extract_positions_list(pos)
            snap.open_positions_count = len(positions)
            snap.open_exposure_dollars = min(
                self.settings.MAX_OPEN_EXPOSURE_DOLLARS,
                len(positions) * self.settings.MIN_TRADE_DOLLARS,
            )
        except Exception as e:
            self.logger.warning(f"Positions fetch failed: {e}")

        snap.daily_pnl_dollars = 0.0
        return snap

    # -----------------------------
    # Position / exit helpers
    # -----------------------------
    def _to_float(self, v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return None
            return float(v)
        except Exception:
            return None

    def _extract_positions_list(self, payload: Any) -> list[dict]:
        """
        Normalize multiple possible position payload layouts from client/API.
        """
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
        no_qty = self._to_float(p.get("no_count") or p.get("no_position") or p.get("no_contracts"))
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

    def _synthetic_open_positions_from_decisions(self) -> list[dict]:
        """
        DRY_RUN only: Builds a list of all unresolved synthetic entries.
        """
        if not bool(self.settings.DRY_RUN):
            return []

        try:
            rows = self.store.recent_decisions(1000)
        except Exception:
            return []

        open_pos = {}
        
        # Reverse to replay state oldest to newest
        for r in reversed(rows):
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue

            ticker = str(r.get("ticker", ""))
            if not ticker:
                continue

            action = str(r.get("action", "") or "")
            reasons = str(r.get("reasons", "") or "")

            if action.startswith("TRADE_EXIT"):
                exit_finalized = bool(r.get("resolved_ts")) or ("exit_consumed:1" in reasons)
                if ticker in open_pos:
                    if exit_finalized:
                        del open_pos[ticker]
                    else:
                        open_pos[ticker]["_exit_pending"] = True
                continue

            if action in {"TRADE_YES", "TRADE_NO"}:
                if r.get("resolved_ts") or "exit_resolved:1" in reasons or "mtm_label:" in reasons:
                    if ticker in open_pos:
                        del open_pos[ticker]
                    continue

                entry_px = None
                count = None
                for part in reasons.split(";"):
                    part = part.strip()
                    if part.startswith("entry_price_cents:"):
                        try:
                            entry_px = int(float(part.split(":", 1)[1].strip()))
                        except Exception:
                            pass
                    elif part.startswith("order_count:"):
                        try:
                            count = int(float(part.split(":", 1)[1].strip()))
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
                    "count": int(max(1, count)),
                    "entry_price_cents": entry_px,
                    "opened_at": r.get("ts"),
                    "entry_id": entry_id,
                    "_synthetic_from_decisions": True,
                    "_exit_pending": False
                }

        return list(open_pos.values())

    def _synthetic_open_position_from_decisions(self, ticker: str) -> Optional[dict]:
        """
        DRY_RUN fallback only: Single ticker lookup for existing flows.
        """
        for pos in self._synthetic_open_positions_from_decisions():
            if pos["ticker"] == ticker:
                return pos
        return None

    def _find_open_position_for_ticker(self, ticker: str) -> Optional[dict]:
        try:
            raw = self.client.get_positions()
            positions = self._extract_positions_list(raw)
        except Exception as e:
            self.logger.warning(f"Position scan for exits failed: {e}")
            positions = []

        for p in positions:
            if self._position_ticker(p) != ticker:
                continue
            count = self._position_count(p)
            side = self._position_contract_side(p)
            if count > 0 and side in {"YES", "NO"}:
                return p

        # DRY_RUN fallback if API doesn't expose simulated positions
        return self._synthetic_open_position_from_decisions(ticker)

    def _latest_unresolved_entry_id_for_ticker(self, ticker: str) -> Optional[int]:
        try:
            rows = self.store.recent_decisions(250)
        except Exception:
            return None

        for r in rows:
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            if str(r.get("ticker", "")) != str(ticker):
                continue
            
            action = str(r.get("action", ""))
            if action not in {"TRADE_YES", "TRADE_NO"}:
                continue

            if r.get("resolved_ts"):
                continue
            
            reasons = str(r.get("reasons", "") or "")
            if "exit_resolved:1" in reasons:
                continue

            entry_id = r.get("id")
            if entry_id is not None:
                try:
                    return int(entry_id)
                except Exception:
                    pass
        return None

    def _latest_entry_context_for_ticker(self, ticker: str) -> dict:
        out = {
            "entry_conviction": None,
            "entry_ts": None,
            "entry_price_cents": None,
            "entry_action": None,
        }
        try:
            rows = self.store.recent_decisions(250)
        except Exception:
            return out

        for r in rows:
            if bool(r.get("dry_run", True)) != bool(self.settings.DRY_RUN):
                continue
            if str(r.get("ticker", "")) != str(ticker):
                continue
            action = str(r.get("action", ""))
            if not action.startswith("TRADE_"):
                continue
            if action.startswith("TRADE_EXIT"):
                continue
            out["entry_conviction"] = self._to_float(r.get("final_conviction"))
            out["entry_ts"] = r.get("ts")
            out["entry_action"] = action
            reasons = str(r.get("reasons", "") or "")
            marker = "entry_price_cents:"
            idx = reasons.find(marker)
            if idx >= 0:
                tail = reasons[idx + len(marker):]
                val = tail.split(";", 1)[0].strip()
                try:
                    out["entry_price_cents"] = int(float(val))
                except Exception:
                    pass
            return out
        return out

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

    def _build_conservative_exit_order(self, chosen: dict, pos: dict) -> Optional[dict]:
        side = self._position_contract_side(pos)
        count = self._position_count(pos)
        if side not in {"YES", "NO"} or count <= 0:
            return None

        yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
        yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
        no_bid = self._to_float(chosen.get("no_bid")) or 0.0
        mid = self._to_float(chosen.get("mid_yes_cents")) or 50.0

        if side == "YES":
            px = int(round(yes_bid)) if yes_bid > 0 else int(max(1, min(99, round(mid) - 1)))
            yes_no = "yes"
        else:
            if no_bid > 0:
                px = int(round(no_bid))
            elif yes_ask > 0:
                px = int(max(1, min(99, round(100 - yes_ask))))
            else:
                inferred_no_mid = 100 - mid
                px = int(max(1, min(99, round(inferred_no_mid) - 1)))
            yes_no = "no"

        return {
            "ticker": chosen["ticker"],
            "side": "sell",
            "yes_no": yes_no,
            "count": int(count),
            "price_cents": int(max(1, min(99, px))),
            "client_order_id": f"exit-{chosen['ticker']}-{int(dt.datetime.now(dt.UTC).timestamp())}",
            "held_side": side,
        }

    def _should_exit_on_thesis_deterioration(
        self,
        *,
        chosen: dict,
        sig: Any,
        final_conviction: float,
        news: dict,
        news_applied: dict,
    ) -> tuple[bool, list[str]]:
        if not self.exit_management_enabled:
            return False, ["exit_mgmt_disabled"]

        pos = self._find_open_position_for_ticker(str(chosen.get("ticker")))
        if not pos:
            return False, ["no_open_position_for_ticker"]

        held_side = self._position_contract_side(pos)
        held_count = self._position_count(pos)
        if held_side not in {"YES", "NO"} or held_count <= 0:
            return False, ["position_unparseable"]

        held_minutes: Optional[float] = None
        p_opened = self._position_open_ts(pos)
        if p_opened is not None:
            held_minutes = max(0.0, (dt.datetime.now(dt.UTC) - p_opened).total_seconds() / 60.0)
        else:
            entry_ctx = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
            held_minutes = self._minutes_since(entry_ctx.get("entry_ts"))

        reasons: list[str] = [f"exit_check_held_side:{held_side}", f"exit_check_count:{held_count}"]

        if held_minutes is not None:
            reasons.append(f"exit_check_held_mins:{held_minutes:.1f}")
            if held_minutes < float(self.exit_min_hold_minutes):
                reasons.append("exit_hold_guard")
                return False, reasons

        sig_dir = str(getattr(sig, "direction", "SKIP") or "SKIP").upper()
        final_conv = float(final_conviction)

        entry_ctx = self._latest_entry_context_for_ticker(str(chosen.get("ticker")))
        entry_conv = self._to_float(entry_ctx.get("entry_conviction"))
        if entry_conv is not None:
            reasons.append(f"exit_entry_conv:{entry_conv:.1f}")
            reasons.append(f"exit_conv_drop:{(entry_conv - final_conv):.1f}")

        opposite_signal = (held_side == "YES" and sig_dir == "NO") or (held_side == "NO" and sig_dir == "YES")
        weak_or_skip = (sig_dir == "SKIP") and (final_conv <= float(self.exit_max_hold_conviction))
        conviction_collapsed = (
            entry_conv is not None and (entry_conv - final_conv) >= float(self.exit_conviction_drop_points)
        )

        news_conf = float(news.get("confidence", 0.0) or 0.0)
        eff_news = float(news_applied.get("effective_news_score", 0.0) or 0.0)
        strong_adverse_news = (
            (held_side == "YES" and eff_news <= -float(self.exit_news_conflict_min_effect))
            or (held_side == "NO" and eff_news >= float(self.exit_news_conflict_min_effect))
        ) and news_conf >= float(self.exit_news_conflict_min_conf)

        reasons.extend([
            f"exit_sig_dir:{sig_dir}",
            f"exit_final_conv:{final_conv:.1f}",
            f"exit_opp_sig:{1 if opposite_signal else 0}",
            f"exit_weak_or_skip:{1 if weak_or_skip else 0}",
            f"exit_conv_collapse:{1 if conviction_collapsed else 0}",
            f"exit_strong_adverse_news:{1 if strong_adverse_news else 0}",
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

    # -----------------------------
    # DRY_RUN pipeline validation helpers
    # -----------------------------
    def _should_force_pipeline_test_entry(self, *, sig: Any, chosen: dict) -> bool:
        if not bool(self.settings.DRY_RUN):
            return False
        if not self.pipeline_test_entries_enabled:
            return False
        if self.pipeline_test_max_entries <= 0:
            return False
        if self._pipeline_test_entries_sent >= int(self.pipeline_test_max_entries):
            return False

        if str(getattr(sig, "direction", "SKIP") or "SKIP").upper() != "SKIP":
            return False

        try:
            existing = self._find_open_position_for_ticker(str(chosen.get("ticker")))
            if existing:
                return False
        except Exception:
            pass

        if self._pipeline_test_last_entry_tick is not None:
            delta = int(self._tick_count) - int(self._pipeline_test_last_entry_tick)
            if delta < int(self.pipeline_test_min_ticks_between):
                return False

        return True

    def _build_pipeline_test_order(self, chosen: dict) -> Optional[dict]:
        if not bool(self.settings.DRY_RUN):
            return None

        yes_ask = self._to_float(chosen.get("yes_ask")) or 0.0
        yes_bid = self._to_float(chosen.get("yes_bid")) or 0.0
        mid = self._to_float(chosen.get("mid_yes_cents")) or 50.0

        if yes_ask > 0:
            px = int(round(yes_ask))
        elif yes_bid > 0:
            px = int(min(99, max(1, round(yes_bid + 1))))
        else:
            px = int(min(99, max(1, round(mid))))

        count = 1

        return {
            "ticker": str(chosen["ticker"]),
            "side": "buy",
            "yes_no": "yes",
            "count": int(count),
            "price_cents": int(max(1, min(99, px))),
            "client_order_id": f"ptest-{chosen['ticker']}-{int(dt.datetime.now(dt.UTC).timestamp())}",
        }

    def _do_dry_run_exit_test_pass(self, now: str) -> None:
        """
        DRY_RUN-only forced exits pass for any unresolved synthetic entries.
        """
        if not self.settings.DRY_RUN:
            return

        open_positions = self._synthetic_open_positions_from_decisions()
        if not open_positions:
            return

        processed = 0
        for pos in open_positions:
            if processed >= 2:  # Process up to 2 positions per tick to avoid flooding
                break
                
            if pos.get("_exit_pending"):
                continue

            mins = self._minutes_since(pos.get("opened_at"))
            if mins is None or mins < 2.0:
                continue

            ticker = pos["ticker"]
            
            # Fetch fresh quote
            m = None
            try:
                # Assuming base client supports get_market by ticker natively
                raw = getattr(self.client, "get_market", lambda t: None)(ticker)
                if raw:
                    m = summarize_market(raw)
            except Exception:
                pass
                
            if not m:
                # Fallback to get_markets array
                try:
                    raw_list = self.client.get_markets(limit=50)
                    for x in extract_markets(raw_list):
                        if str(x.get("ticker")) == ticker:
                            m = summarize_market(x)
                            break
                except Exception:
                    pass

            if not m:
                continue
                
            # Build basic reverse order
            side = pos["yes_no"]
            yes_bid = self._to_float(m.get("yes_bid")) or 0.0
            yes_ask = self._to_float(m.get("yes_ask")) or 0.0
            no_bid = self._to_float(m.get("no_bid")) or 0.0
            mid = self._to_float(m.get("mid_yes_cents")) or 50.0

            if side == "YES":
                px = int(round(yes_bid)) if yes_bid > 0 else int(max(1, min(99, round(mid) - 1)))
                yes_no = "yes"
            else:
                if no_bid > 0:
                    px = int(round(no_bid))
                elif yes_ask > 0:
                    px = int(max(1, min(99, round(100 - yes_ask))))
                else:
                    px = int(max(1, min(99, round((100 - mid) - 1))))
                yes_no = "no"

            exit_order = {
                "ticker": ticker,
                "side": "sell",
                "yes_no": yes_no,
                "count": int(pos["count"]),
                "price_cents": int(max(1, min(99, px))),
                "client_order_id": f"xtest-{ticker}-{int(dt.datetime.now(dt.UTC).timestamp())}",
            }

            try:
                self.client.place_order(**exit_order)
                action = f"TRADE_EXIT_{side}"
                self.logger.info(f"DRY_RUN forced exit test fired for {ticker} (entry_id: {pos.get('entry_id')})")
            except KalshiAPIError as e:
                action = "ERROR"
                self.logger.warning(f"Exit test order error for {ticker}: {e}")

            entry_id = pos.get("entry_id")
            reasons = [
                "exit_test:1",
                f"exit_price_cents:{exit_order['price_cents']}",
                f"exit_order_side:{exit_order['side']}",
                f"exit_yes_no:{exit_order['yes_no']}",
                f"exit_count:{exit_order['count']}",
            ]
            if entry_id is not None:
                reasons.append(f"entry_id:{entry_id}")

            row = {
                "ts": now,
                "ticker": ticker,
                "direction": "SKIP",
                "base_conviction": 0.0,
                "social_bonus": 0.0,
                "final_conviction": 0.0,
                "stake_dollars": 0.0,
                "action": action,
                "reasons": ";".join(reasons),
                "dry_run": self.settings.DRY_RUN,
                "realized_pnl": None,
                "won": None,
                "resolved_ts": None,
                "market_category": m.get("category"),
            }
            self._record_decision_row(row)
            processed += 1

    def _record_decision_row(self, row: dict) -> Optional[int]:
        inserted_id: Optional[int] = None
        try:
            inserted_id = self.store.insert_decision(row)
        except TypeError:
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
            inserted_id = self.store.insert_decision(legacy_row)

            try:
                if hasattr(self.store, "update_decision_metadata") and inserted_id is not None:
                    self.store.update_decision_metadata(
                        decision_id=int(inserted_id),
                        updates={
                            "news_score": row.get("news_score"),
                            "news_confidence": row.get("news_confidence"),
                            "news_regime": row.get("news_regime"),
                            "news_effective_score": row.get("news_effective_score"),
                            "spread_cents": row.get("spread_cents"),
                        },
                    )
            except Exception as e:
                self.logger.warning(f"Post-insert metadata patch failed (safe to ignore): {e}")
        return inserted_id

    def _post_tick_housekeeping(self, snapshot: PortfolioSnapshot) -> None:
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

    def _maybe_manage_exit_for_chosen(
        self,
        *,
        now: str,
        chosen: dict,
        sig: Any,
        social: Any,
        news: dict,
        news_applied: dict,
        final_conviction: float,
    ) -> bool:
        should_exit, exit_reasons = self._should_exit_on_thesis_deterioration(
            chosen=chosen,
            sig=sig,
            final_conviction=final_conviction,
            news=news,
            news_applied=news_applied,
        )
        if not should_exit:
            return False

        pos = self._find_open_position_for_ticker(str(chosen.get("ticker")))
        if not pos:
            return False

        if pos.get("_exit_pending"):
            self.logger.info(f"EXIT skip {chosen['ticker']} | pending exit order is already present")
            return False

        exit_order = self._build_conservative_exit_order(chosen, pos)
        if not exit_order:
            self.logger.info(f"EXIT skip {chosen['ticker']} | could not build exit order from position payload")
            return False

        entry_id = None
        if pos.get("_synthetic_from_decisions") and pos.get("entry_id") is not None:
            entry_id = pos.get("entry_id")
        else:
            entry_id = self._latest_unresolved_entry_id_for_ticker(str(chosen.get("ticker")))

        exit_action = f"TRADE_EXIT_{exit_order['held_side']}"
        reasons = list(getattr(sig, "reasons", [])) + [
            f"social:{','.join(getattr(social, 'reasons', []))}",
            "risk:exit_thesis_deterioration",
            f"news:{news.get('regime', 'unavailable')}",
            f"news_score_raw:{float(news.get('score', 0.0)):.2f}",
            f"news_score_eff:{float(news_applied.get('effective_news_score', 0.0)):.2f}",
            f"news_conf:{float(news.get('confidence', 0.0)):.2f}",
            f"exit_price_cents:{int(exit_order['price_cents'])}",
            f"exit_order_side:{exit_order['side']}",
            f"exit_yes_no:{exit_order['yes_no']}",
            f"exit_count:{int(exit_order['count'])}",
        ] + exit_reasons

        if entry_id is not None:
            reasons.append(f"entry_id:{entry_id}")

        action = "SKIP"
        try:
            self.client.place_order(
                ticker=exit_order["ticker"],
                side=exit_order["side"],
                yes_no=exit_order["yes_no"],
                count=exit_order["count"],
                price_cents=exit_order["price_cents"],
                client_order_id=exit_order["client_order_id"],
            )
            action = exit_action
            self.logger.info(
                f"{exit_action} {chosen['ticker']} count={exit_order['count']} "
                f"price={exit_order['price_cents']}c dry_run={self.settings.DRY_RUN}"
            )
            self.notifier.send(
                f"📉 {exit_action} `{chosen['ticker']}` | count={exit_order['count']} | px={exit_order['price_cents']}c "
                f"| conv={final_conviction:.1f} | news_eff={float(news_applied.get('effective_news_score', 0.0)):.1f} "
                f"| dry_run={self.settings.DRY_RUN}"
            )
        except KalshiAPIError as e:
            action = "ERROR"
            self.logger.exception(f"Exit order error: {e}")
            self.notifier.send(f"⚠️ Kalshi exit order error: {e}")

        row = {
            "ts": now,
            "ticker": chosen["ticker"],
            "direction": str(getattr(sig, "direction", "SKIP")),
            "base_conviction": float(getattr(sig, "conviction_score", 0.0)),
            "social_bonus": float(getattr(social, "bonus_score", 0.0)),
            "final_conviction": float(final_conviction),
            "stake_dollars": 0.0,
            "action": action,
            "reasons": ";".join(reasons),
            "dry_run": self.settings.DRY_RUN,
            "realized_pnl": None,
            "won": None,
            "resolved_ts": None,
            "market_category": chosen.get("category"),
            "news_score": float(news.get("score", 0.0)),
            "news_confidence": float(news.get("confidence", 0.0)),
            "news_regime": str(news.get("regime", "unavailable")),
            "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
            "spread_cents": chosen.get("spread_cents"),
        }
        self._record_decision_row(row)
        return True

    def _fetch_market_universe(self, pages: int = 5, page_limit: int = 100) -> list[dict]:
        cursor: Optional[str] = None
        pages_fetched = 0

        total_seen = 0
        dropped_mve = 0
        dropped_provisional = 0
        duplicate_tickers = 0
        dropped_preferred_quote = 0

        seen_tickers: set[str] = set()
        preferred_non_mve: list[dict] = []
        mve_fallback_candidates: list[dict] = []

        fallback_min_mid = 3.0
        fallback_max_mid = 97.0
        fallback_min_volume = 0.0
        fallback_require_at_least_one_side = True

        for _ in range(max(1, int(pages))):
            try:
                raw = self.client.get_markets(limit=int(page_limit), cursor=cursor)
            except TypeError:
                raw = self.client.get_markets(limit=int(page_limit))

            page_markets = extract_markets(raw)
            pages_fetched += 1

            if not page_markets:
                break

            for m in page_markets:
                ticker = str(m.get("ticker", "") or "").strip()
                if not ticker:
                    continue

                if ticker in seen_tickers:
                    duplicate_tickers += 1
                    continue
                seen_tickers.add(ticker)
                total_seen += 1

                is_mve = ticker.startswith("KXMVESPORTSMULTIGAMEEXTENDED") or bool(m.get("mve_collection_ticker"))
                is_provisional = bool(m.get("is_provisional"))

                try:
                    s = summarize_market(m)
                except Exception:
                    continue

                status_ok = str(s.get("status", "")).lower() in {"active", "open", "trading", "unknown"}
                spread = float(s.get("spread_cents", 0) or 0)
                mid = float(s.get("mid_yes_cents", 0) or 0)
                vol = float(s.get("volume", 0) or 0)
                yes_bid = float(s.get("yes_bid", 0) or 0)
                yes_ask = float(s.get("yes_ask", 0) or 0)

                quote_ok_standard = (
                    status_ok
                    and spread > 0
                    and spread <= int(self.settings.MAX_SPREAD_CENTS)
                    and mid > 2
                    and mid < 98
                    and vol >= float(self.settings.MIN_RECENT_VOLUME)
                )

                one_side_ok = (yes_bid > 0 or yes_ask > 0) if fallback_require_at_least_one_side else True
                quote_ok_for_mve_fallback = (
                    status_ok
                    and spread > 0
                    and spread <= int(self.settings.MAX_SPREAD_CENTS)
                    and fallback_min_mid <= mid <= fallback_max_mid
                    and one_side_ok
                    and vol >= fallback_min_volume
                )

                if is_mve:
                    dropped_mve += 1
                    if quote_ok_for_mve_fallback:
                        mve_fallback_candidates.append(m)
                    continue

                if is_provisional:
                    dropped_provisional += 1
                    continue

                if quote_ok_standard:
                    preferred_non_mve.append(m)
                else:
                    dropped_preferred_quote += 1

            if not isinstance(raw, dict):
                break
            cursor = raw.get("cursor")
            if not cursor:
                break

            if len(preferred_non_mve) >= 150:
                break

        if preferred_non_mve:
            self.logger.info(
                "Universe fetch | seen=%s kept=%s dropped_mve=%s dropped_provisional=%s "
                "dropped_preferred_quote=%s dupes=%s fallback_mve_candidates=%s pages=%s",
                total_seen,
                len(preferred_non_mve),
                dropped_mve,
                dropped_provisional,
                dropped_preferred_quote,
                duplicate_tickers,
                len(mve_fallback_candidates),
                pages_fetched,
            )
            return preferred_non_mve

        if mve_fallback_candidates:
            ranked = []
            for m in mve_fallback_candidates:
                try:
                    ranked.append(summarize_market(m))
                except Exception:
                    continue

            ranked.sort(key=lambda x: (x["spread_cents"], abs(x["mid_yes_cents"] - 50), -x["volume"]))
            fallback_markets = [r.get("raw", {}) for r in ranked if r.get("raw")][:150]

            self.logger.warning(
                "Universe fetch fallback ACTIVE | seen=%s kept_non_mve=0 dropped_mve=%s dropped_provisional=%s "
                "dropped_preferred_quote=%s dupes=%s mve_fallback_kept=%s pages=%s "
                "| fallback_rules: one_side=%s mid=[%.1f,%.1f] vol>=%.1f spread<=%s",
                total_seen,
                dropped_mve,
                dropped_provisional,
                dropped_preferred_quote,
                duplicate_tickers,
                len(fallback_markets),
                pages_fetched,
                fallback_require_at_least_one_side,
                fallback_min_mid,
                fallback_max_mid,
                fallback_min_volume,
                int(self.settings.MAX_SPREAD_CENTS),
            )
            return fallback_markets

        self.logger.warning(
            "Universe fetch | seen=%s kept=0 dropped_mve=%s dropped_provisional=%s "
            "dropped_preferred_quote=%s dupes=%s mve_fallback_kept=0 pages=%s",
            total_seen,
            dropped_mve,
            dropped_provisional,
            dropped_preferred_quote,
            duplicate_tickers,
            pages_fetched,
        )
        return []

    def _choose_market(self, markets: list[dict]) -> Any:
        ok_status = {"active", "open", "trading", "unknown"}

        strict_candidates: list[dict] = []
        for m in markets[:200]:
            try:
                s = summarize_market(m)
            except Exception:
                continue

            if str(s.get("status", "")).lower() not in ok_status:
                continue

            spread = float(s.get("spread_cents", 0) or 0)
            volume = float(s.get("volume", 0) or 0)
            mid = float(s.get("mid_yes_cents", 0) or 0)

            if spread <= 0 or spread > float(self.settings.MAX_SPREAD_CENTS):
                continue
            if volume < float(self.settings.MIN_RECENT_VOLUME):
                continue
            if mid <= 2 or mid >= 98:
                continue

            s["_pick_mode"] = "strict"
            s["_effective_spread"] = spread
            strict_candidates.append(s)

        strict_candidates.sort(
            key=lambda x: (
                x.get("_effective_spread", 999),
                abs((x.get("mid_yes_cents", 50) or 50) - 50),
                -(x.get("volume", 0) or 0),
            )
        )
        if strict_candidates:
            return strict_candidates[0]

        relaxed_candidates: list[dict] = []
        max_relaxed_spread = max(
            float(self.settings.MAX_SPREAD_CENTS) * 2.0,
            float(self.settings.MAX_SPREAD_CENTS) + 5.0,
        )

        relaxed_mid_floor = 15.0
        relaxed_mid_ceiling = 85.0

        for m in markets[:200]:
            try:
                s = summarize_market(m)
            except Exception:
                continue

            if str(s.get("status", "")).lower() not in ok_status:
                continue

            spread = float(s.get("spread_cents", 0) or 0)
            mid = float(s.get("mid_yes_cents", 0) or 0)
            volume = float(s.get("volume", 0) or 0)
            yes_bid = float(s.get("yes_bid", 0) or 0)
            yes_ask = float(s.get("yes_ask", 0) or 0)

            if yes_bid <= 0 and yes_ask <= 0 and mid <= 0:
                continue
            if mid and (mid <= 1 or mid >= 99):
                continue
            if mid and (mid < relaxed_mid_floor or mid > relaxed_mid_ceiling):
                continue

            if spread > 0:
                effective_spread = spread
            elif yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
                effective_spread = yes_ask - yes_bid
            else:
                effective_spread = 99.0

            if effective_spread > max_relaxed_spread:
                continue

            s["_pick_mode"] = "relaxed"
            s["_effective_spread"] = effective_spread
            s["_relaxed"] = True
            s["_relaxed_mid_guard"] = f"{relaxed_mid_floor:.0f}-{relaxed_mid_ceiling:.0f}"
            s["_volume_for_sort"] = volume
            relaxed_candidates.append(s)

        relaxed_candidates.sort(
            key=lambda x: (
                x.get("_effective_spread", 999),
                abs((x.get("mid_yes_cents", 50) or 50) - 50),
                -(x.get("_volume_for_sort", 0) or 0),
            )
        )
        return relaxed_candidates[0] if relaxed_candidates else None

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

    def _news_category_allowlist(self) -> list[str]:
        return ["politics", "weather", "sports", "macro"]

    def _fetch_news_items_for_market(self, chosen: dict) -> list[dict]:
        try:
            from app.news_fetch import (
                build_news_query_from_market_label,
                fetch_google_news_rss,
                infer_direction_hint,
                yes_means_event_occurs,
            )
        except Exception as e:
            self.logger.warning(f"News fetch module unavailable: {e}")
            return []

        label = str(
            chosen.get("market_label")
            or chosen.get("title")
            or chosen.get("question")
            or chosen.get("name")
            or ""
        ).strip()

        if not label:
            label = str(chosen.get("ticker") or "").strip()
        if not label:
            return []

        query = build_news_query_from_market_label(label)
        if not query:
            return []
        
        self.logger.info(f"News query='{query}' | label='{label[:120]}' | ticker={chosen.get('ticker')}")
        
        cache = getattr(self, "news_cache", None)

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
        raw_news_score = float(news.get("score", 0.0))
        news_conf = max(0.0, min(1.0, float(news.get("confidence", 0.0))))
        news_regime = str(news.get("regime", "unavailable"))
        sig_dir = str(base_signal_direction or "SKIP").upper()

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
        elif news_regime in {"mixed"}:
            regime_scale = self.news_mixed_regime_multiplier

        conflict = False
        if sig_dir == "YES" and raw_news_score < 0:
            conflict = True
        elif sig_dir == "NO" and raw_news_score > 0:
            conflict = True

        direction_scale = self.news_conflict_penalty_multiplier if conflict else 1.0

        effective_news_score = raw_news_score * conf_scale * regime_scale * direction_scale

        if (not self.news_allow_direction_flip) and conflict:
            effective_news_score = max(-2.0, min(2.0, effective_news_score))

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

        markets = self._fetch_market_universe(pages=5, page_limit=100)
        self.logger.info(f"Fetched filtered market universe: {len(markets)}")

        if not markets:
            self.logger.warning("Filtered universe empty; falling back to single-page raw markets.")
            raw = self.client.get_markets(limit=100)
            markets = extract_markets(raw)
            self.logger.info(f"Fetched markets (fallback raw): {len(markets)}")

        if not markets:
            return

        chosen = self._choose_market(markets)
        if not chosen:
            self.logger.info("No candidate market found.")
            snapshot = self._fetch_portfolio_snapshot()
            recent = self.store.recent_decisions(80)
            self._maybe_send_periodic_summary(snapshot, recent)
            self._maybe_send_scorecard_summary()
            
            # Allow the global dry-run exit pass even if no current candidate was chosen
            if self.settings.DRY_RUN:
                self._do_dry_run_exit_test_pass(now)
            return

        market_label = str(chosen.get("market_label") or chosen.get("title") or chosen.get("question") or "").strip()
        if not market_label:
            market_label = str(chosen.get("ticker") or "")

        mid_yes = float(chosen.get("mid_yes_cents", 0.0) or 0.0)
        mid_no = 100.0 - mid_yes if mid_yes else 0.0

        self.logger.info(
            f"Candidate {chosen['ticker']} | mid_yes={mid_yes:.1f} mid_no={mid_no:.1f} "
            f"| spread={chosen.get('spread_cents')} | vol={chosen.get('volume')} | pick={chosen.get('_pick_mode','')}"
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

        try:
            exited = self._maybe_manage_exit_for_chosen(
                now=now,
                chosen=chosen,
                sig=sig,
                social=social,
                news=news,
                news_applied=news_applied,
                final_conviction=final_conviction,
            )
            if exited:
                snapshot = self._fetch_portfolio_snapshot()
                
                # Allow the global dry-run exit pass
                if self.settings.DRY_RUN:
                    self._do_dry_run_exit_test_pass(now)
                
                self._post_tick_housekeeping(snapshot)
                return
        except Exception as e:
            self.logger.warning(f"Exit-management check failed (continuing to entry logic): {e}")

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
            available_cash_dollars=snapshot.cash_balance_dollars,
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
        
        try:
            my = float(chosen.get("mid_yes_cents", 0.0) or 0.0)
            reasons.append(f"mid_yes_cents:{my:.2f}")
            reasons.append(f"mid_no_cents:{(100.0-my):.2f}")
        except Exception:
            pass

        if market_label:
            reasons.append(f"market_label:{market_label[:140]}")

        if news.get("risk_flags"):
            reasons.append(f"news_flags:{','.join(news['risk_flags'])}")

        if recent_win_rate_for_gate is not None:
            reasons.append(f"risk_gate_wr:{recent_win_rate_for_gate:.3f}")
            reasons.append(f"risk_gate_wr_n:{winrate_sample}")

        force_guard_skip = False
        if getattr(sig, "direction", "SKIP") != "SKIP":
            if self._find_open_position_for_ticker(str(chosen["ticker"])):
                self.logger.info(f"Guard: Open position already exists for {chosen['ticker']}. Forcing SKIP.")
                force_guard_skip = True
                reasons.append("guard:already_have_open_position")
                try:
                    sig.direction = "SKIP"
                except AttributeError:
                    pass

        forced_pipeline_test = False
        pipeline_test_order = None

        try:
            if self._should_force_pipeline_test_entry(sig=sig, chosen=chosen):
                pipeline_test_order = self._build_pipeline_test_order(chosen)
                if pipeline_test_order is not None:
                    forced_pipeline_test = True
                    reasons.append("pipeline_test_entry:1")
                    reasons.append("pipeline_test_scope:dry_run_only")
                    self.logger.info(
                        f"DRY_RUN pipeline test entry armed for {chosen['ticker']} "
                        f"(count={pipeline_test_order.get('count')} px={pipeline_test_order.get('price_cents')})"
                    )
        except Exception as e:
            self.logger.warning(f"Pipeline test entry check failed (continuing normal logic): {e}")

        should_enter_normal = (sig.direction != "SKIP" and not force_guard_skip and risk.allowed)
        should_enter_test = bool(forced_pipeline_test and pipeline_test_order is not None)

        if should_enter_normal or should_enter_test:
            if should_enter_test:
                order = pipeline_test_order
                used_direction = "YES"
                used_stake_dollars = float(max(1.0, self.settings.MIN_TRADE_DOLLARS))
                reasons.append("risk:pipeline_test_bypass")
            else:
                order = build_limit_order_from_signal(chosen["ticker"], sig.direction, chosen, risk.stake_dollars)
                used_direction = str(sig.direction)
                used_stake_dollars = float(risk.stake_dollars)

            order_price_cents: Optional[int] = None
            order_count: Optional[int] = None

            try:
                order_price_cents = int(order.get("price_cents"))  # type: ignore[arg-type]
            except Exception:
                order_price_cents = None

            try:
                order_count = int(order.get("count"))  # type: ignore[arg-type]
            except Exception:
                order_count = None

            if order_price_cents is not None:
                order_price_cents = int(max(1, min(99, order_price_cents)))
                reasons.append(f"entry_price_cents:{order_price_cents}")
            else:
                reasons.append("entry_price_cents:missing")

            if order_count is not None:
                order_count = int(max(0, order_count))
                reasons.append(f"order_count:{order_count}")
            else:
                reasons.append("order_count:missing")

            try:
                reasons.append(f"order_side:{str(order.get('side'))}")
                reasons.append(f"order_yes_no:{str(order.get('yes_no'))}")
            except Exception:
                pass

            try:
                self.client.place_order(
                    ticker=order["ticker"],
                    side=order["side"],
                    yes_no=order["yes_no"],
                    count=order["count"],
                    price_cents=order["price_cents"],
                    client_order_id=order["client_order_id"],
                )
                action = f"TRADE_{used_direction}"
                self.logger.info(
                    f"{action} {chosen['ticker']} stake=${used_stake_dollars:.2f} "
                    f"price={order['price_cents']}c count={order['count']} dry_run={self.settings.DRY_RUN}"
                )

                if should_enter_test:
                    self._pipeline_test_entries_sent += 1
                    self._pipeline_test_last_entry_tick = self._tick_count

                self.notifier.send(
                    f"📈 {action} `{chosen['ticker']}` | stake=${used_stake_dollars:.2f} "
                    f"| conv={final_conviction:.1f} "
                    f"(base {sig.conviction_score:.1f} + social {social.bonus_score:.1f} + news_eff {float(news_applied['effective_news_score']):.1f}) "
                    f"| learning={learning['mode']} x{float(learning['stake_multiplier']):.2f} "
                    f"| news={news['regime']} conf={float(news.get('confidence', 0.0)):.2f} "
                    f"| gate_wr={('n/a' if recent_win_rate_for_gate is None else f'{recent_win_rate_for_gate*100:.1f}%')} "
                    f"| dry_run={self.settings.DRY_RUN}"
                    f"{' | pipeline_test=1' if should_enter_test else ''}"
                )

            except KalshiAPIError as e:
                action = "ERROR"
                self.logger.exception(f"Order error: {e}")
                self.notifier.send(f"⚠️ Kalshi order error: {e}")

        else:
            self.logger.info(
                f"SKIP {chosen['ticker']} | dir={sig.direction} | conv={final_conviction:.1f} | risk={risk.reason}"
            )

        joined_reasons = ";".join(reasons)

        row = {
            "ts": now,
            "ticker": chosen["ticker"],
            "direction": sig.direction,
            "base_conviction": sig.conviction_score,
            "social_bonus": social.bonus_score,
            "final_conviction": final_conviction,
            "stake_dollars": (
                float(max(1.0, self.settings.MIN_TRADE_DOLLARS))
                if (action.startswith("TRADE_") and "pipeline_test_entry:1" in joined_reasons)
                else (risk.stake_dollars if action.startswith("TRADE_") else 0.0)
            ),
            "action": action,
            "reasons": joined_reasons,
            "dry_run": self.settings.DRY_RUN,
            "realized_pnl": None,
            "won": None,
            "resolved_ts": None,
            "market_category": chosen.get("category"),
            "news_score": float(news.get("score", 0.0)),
            "news_confidence": float(news.get("confidence", 0.0)),
            "news_regime": str(news.get("regime", "unavailable")),
            "news_effective_score": float(news_applied.get("effective_news_score", 0.0)),
            "spread_cents": chosen.get("spread_cents"),
        }

        self._record_decision_row(row)
        
        # -----------------------------
        # Forced DRY_RUN exit pass for any unresolved synthetic positions
        # -----------------------------
        if self.settings.DRY_RUN:
            self._do_dry_run_exit_test_pass(now)
            
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
