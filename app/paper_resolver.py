from __future__ import annotations

import datetime as dt
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from app.market_data import summarize_market


@dataclass
class PaperResolutionResult:
    updated: int
    checked: int
    skipped: int


# Backward/alt naming safety (in case other code imported old class name)
PaperResolveResult = PaperResolutionResult


_RE_NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def _parse_iso(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.UTC)
        return d.astimezone(dt.UTC)
    except Exception:
        return None


def _get_db_path_from_store(store: Any) -> str:
    """
    Best-effort DB path discovery.
    Falls back to 'bot_state.db' which matches your Render shell audits.
    """
    for attr in ("DB_PATH", "db_path", "path", "_db_path"):
        p = getattr(store, attr, None)
        if isinstance(p, str) and p.strip():
            return p.strip()
    return "bot_state.db"


def _parse_number_after(prefix: str, reasons: str) -> Optional[float]:
    """
    Robustly parse number after e.g. 'entry_price_cents:' even if:
      - it's the last token (no trailing ';')
      - reasons contains newlines
      - token has spaces
    """
    if not reasons:
        return None
    s = str(reasons)
    idx = s.find(prefix)
    if idx < 0:
        return None
    tail = s[idx + len(prefix) :]

    semi = tail.find(";")
    if semi >= 0:
        tail = tail[:semi]
    tail = tail.strip()

    m = _RE_NUM.search(tail)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_int_after(prefix: str, reasons: str) -> Optional[int]:
    v = _parse_number_after(prefix, reasons)
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def _clamp_cents(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        i = int(round(float(v)))
        if 1 <= i <= 99:
            return i
        return None
    except Exception:
        return None


def _extract_entry_price_cents(reasons: str) -> Optional[int]:
    # side price (YES if TRADE_YES; NO if TRADE_NO)
    return _clamp_cents(_parse_int_after("entry_price_cents:", reasons or ""))


def _extract_exit_price_cents(reasons: str) -> Optional[int]:
    """
    Exit rows SHOULD log exit_price_cents, but if your exit code logs entry_price_cents:<exit_px>,
    we support both.
    """
    px = _parse_int_after("exit_price_cents:", reasons or "")
    if px is None:
        px = _parse_int_after("entry_price_cents:", reasons or "")
    return _clamp_cents(px)


def _extract_order_count(reasons: str) -> Optional[int]:
    c = _parse_int_after("order_count:", reasons or "")
    if c is None:
        return None
    try:
        c = int(c)
    except Exception:
        return None
    return int(max(0, c))


def _extract_exit_count(reasons: str) -> Optional[int]:
    c = _parse_int_after("exit_count:", reasons or "")
    if c is None:
        return None
    try:
        c = int(c)
    except Exception:
        return None
    return int(max(0, c))


def _compute_realized_pnl_from_prices(*, entry_px: int, exit_px: int, count: int) -> float:
    """
    PnL in *side price space* (YES prices for YES trades, NO prices for NO trades):
      pnl = (exit - entry) * count / 100
    """
    e = int(max(1, min(99, int(entry_px))))
    x = int(max(1, min(99, int(exit_px))))
    c = int(max(0, int(count)))
    if c <= 0:
        return 0.0
    return float(round(((float(x) - float(e)) * float(c) / 100.0), 4))


def _compute_mtm_pnl_from_mark(
    *,
    action: str,
    entry_px_side_cents: int,
    mark_yes_cents: int,
    count: int,
) -> float:
    """
    MTM using *order_count* (NOT stake inference).

    - TRADE_YES: entry_px is YES price; mark uses mid_yes
    - TRADE_NO:  entry_px is NO  price; mark uses mid_no = 100 - mid_yes

    Returns: pnl (float)
    """
    a = str(action or "")
    c = int(max(0, int(count)))
    if c <= 0:
        return 0.0

    mark_yes_i = int(max(1, min(99, int(mark_yes_cents))))

    if a == "TRADE_YES":
        entry_yes = int(max(1, min(99, int(entry_px_side_cents))))
        return _compute_realized_pnl_from_prices(entry_px=entry_yes, exit_px=mark_yes_i, count=c)

    if a == "TRADE_NO":
        mark_no = int(max(1, min(99, 100 - mark_yes_i)))
        entry_no = int(max(1, min(99, int(entry_px_side_cents))))
        return _compute_realized_pnl_from_prices(entry_px=entry_no, exit_px=mark_no, count=c)

    return 0.0


def resolve_paper_trades(
    *,
    store: Any,
    client: Any,
    logger: Any = None,
    max_to_check: int = 25,
    min_age_minutes: int = 15,
) -> PaperResolutionResult:
    """
    DRY_RUN trade lifecycle resolver.

    Pass 1) Resolve explicit exits:
      - Pair TRADE_EXIT_YES/NO to latest unresolved TRADE_YES/NO for same ticker
      - Compute realized PnL using entry/exit prices in side-space and count
      - Tags both entry + exit reasons for auditability:
          entry: exit_resolved:1;exit_id:<id>;exit_px_cents:<px>;exit_cnt:<cnt>;exit_side:<YES|NO>
          exit:  exit_consumed:1;entry_id:<id>;exit_resolve_reason:explicit

    Pass 2) MTM resolve old entries with no exits:
      - Uses current mark from market (prefers mid_yes; else avg(bid,ask); else bid; else ask)
      - Uses entry_price_cents + order_count from reasons
      - NO trades use mark_no = 100 - mark_yes
      - If market fetch fails -> resolves with pnl=0 + mtm_label:market_fetch_failed
      - If quote is unusable -> DELAY policy:
          * if too fresh: defer (leave unresolved)
          * if mid-age: keep deferring (leave unresolved)
          * if very old: resolve to pnl=0 + mtm_label:unusable_quote (so it doesn't stall forever)
      - Adds mtm_label tags so training can skip unlearnable rows
    """
    db_path = _get_db_path_from_store(store)
    now = dt.datetime.now(dt.UTC)
    now_iso = now.isoformat()

    updated = 0
    checked = 0
    skipped = 0

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Ensure table exists
    try:
        cur.execute("SELECT 1 FROM decisions LIMIT 1;").fetchone()
    except Exception as e:
        con.close()
        raise RuntimeError(f"decisions table not found or unreadable in {db_path}: {e}")

    def _append_tag(reasons: str, tag: str) -> str:
        rs = str(reasons or "")
        if not tag:
            return rs
        if tag in rs:
            return rs
        if rs and not rs.endswith(";"):
            rs += ";"
        return rs + tag + ";"

    def _safe_mark_yes_from_summary(s: dict) -> tuple[Optional[int], str]:
        """
        Return (mark_yes_cents, source).
        Prefers mid_yes; falls back to avg(bid,ask); then bid; then ask.
        """
        mid = s.get("mid_yes_cents")
        bid = s.get("yes_bid")
        ask = s.get("yes_ask")

        def _to_int(x) -> Optional[int]:
            try:
                if x is None:
                    return None
                v = int(round(float(x)))
                if 1 <= v <= 99:
                    return v
            except Exception:
                return None
            return None

        mid_i = _to_int(mid)
        if mid_i is not None:
            return mid_i, "mid_yes"

        bid_i = _to_int(bid)
        ask_i = _to_int(ask)

        if bid_i is not None and ask_i is not None and ask_i >= bid_i:
            return int(round((bid_i + ask_i) / 2.0)), "avg_bid_ask"
        if bid_i is not None:
            return bid_i, "bid_only"
        if ask_i is not None:
            return ask_i, "ask_only"
        return None, "no_quote"

    # Learning hygiene: don't label pennies as win/loss
    flat_pnl_epsilon = 0.01

    # Unusable-quote delay policy knobs
    unusable_defer_minutes = 60               # don't label unusable quotes for at least 60 minutes
    unusable_force_resolve_minutes = 12 * 60  # after 12 hours, force resolve to 0.0 (so it can't stall forever)

    # Single safe transaction for both passes
    try:
        try:
            cur.execute("BEGIN;")
        except Exception:
            pass

        # -----------------------------
        # PASS 1: Resolve explicit exits
        # -----------------------------
        exit_rows = cur.execute(
            """
            SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
            FROM decisions
            WHERE dry_run = 1
              AND action IN ('TRADE_EXIT_YES', 'TRADE_EXIT_NO')
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max_to_check),),
        ).fetchall()

        for ex in exit_rows:
            checked += 1

            # already consumed
            if ex["resolved_ts"] is not None:
                skipped += 1
                continue

            ex_ts = _parse_iso(str(ex["ts"] or ""))
            if ex_ts is None:
                skipped += 1
                continue

            age_min = (now - ex_ts).total_seconds() / 60.0
            if age_min < float(min_age_minutes):
                skipped += 1
                continue

            ticker = str(ex["ticker"] or "").strip()
            if not ticker:
                skipped += 1
                continue

            exit_action = str(ex["action"] or "")
            held_side = "YES" if exit_action.endswith("_YES") else "NO"
            entry_action = "TRADE_YES" if held_side == "YES" else "TRADE_NO"

            exit_reasons = str(ex["reasons"] or "")
            exit_px = _extract_exit_price_cents(exit_reasons)
            exit_cnt = _extract_exit_count(exit_reasons)

            if exit_px is None:
                skipped += 1
                continue

            # Find latest unresolved matching entry
            ent = cur.execute(
                """
                SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
                FROM decisions
                WHERE dry_run = 1
                  AND ticker = ?
                  AND action = ?
                  AND resolved_ts IS NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (ticker, entry_action),
            ).fetchone()

            if not ent:
                skipped += 1
                continue

            entry_id = int(ent["id"])
            entry_reasons = str(ent["reasons"] or "")
            entry_px = _extract_entry_price_cents(entry_reasons)
            entry_cnt = _extract_order_count(entry_reasons)

            if entry_px is None or entry_cnt is None or int(entry_cnt) <= 0:
                skipped += 1
                continue

            count_used = int(entry_cnt)
            if exit_cnt is not None and int(exit_cnt) > 0:
                count_used = int(min(count_used, int(exit_cnt)))
            if count_used <= 0:
                skipped += 1
                continue

            pnl = _compute_realized_pnl_from_prices(
                entry_px=int(entry_px),
                exit_px=int(exit_px),
                count=int(count_used),
            )

            max_abs = float(count_used) * 0.99
            if abs(float(pnl)) > (max_abs + 1e-6):
                skipped += 1
                if logger:
                    logger.warning(
                        "PaperResolver (EXIT) sanity-skip "
                        f"entry_id={entry_id} exit_id={int(ex['id'])} ticker={ticker} side={held_side} "
                        f"pnl={float(pnl):.4f} > max_abs={max_abs:.2f} "
                        f"(entry_px={entry_px} exit_px={exit_px} count_used={count_used})"
                    )
                continue

            won_val: int = 1 if float(pnl) > 0 else 0

            # Tag entry + exit
            entry_new_reasons = entry_reasons
            for tg in (
                "exit_resolved:1",
                f"exit_id:{int(ex['id'])}",
                f"exit_px_cents:{int(exit_px)}",
                f"exit_cnt:{int(count_used)}",
                f"exit_side:{held_side}",
            ):
                entry_new_reasons = _append_tag(entry_new_reasons, tg)

            exit_new_reasons = exit_reasons
            for tg in ("exit_consumed:1", f"entry_id:{entry_id}", "exit_resolve_reason:explicit"):
                exit_new_reasons = _append_tag(exit_new_reasons, tg)

            cur.execute(
                """
                UPDATE decisions
                SET realized_pnl = ?,
                    won = ?,
                    resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                """,
                (float(pnl), int(won_val), now_iso, entry_new_reasons, entry_id),
            )

            cur.execute(
                """
                UPDATE decisions
                SET resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                """,
                (now_iso, exit_new_reasons, int(ex["id"])),
            )

            updated += 1

            if logger:
                logger.info(
                    "PaperResolver (EXIT) resolved "
                    f"ticker={ticker} side={held_side} entry_id={entry_id} exit_id={int(ex['id'])} "
                    f"entry_px={int(entry_px)} exit_px={int(exit_px)} count={count_used} "
                    f"pnl={float(pnl):.4f} won={won_val}"
                )

        # -----------------------------
        # PASS 2: MTM resolve old entries (no exit found)
        # -----------------------------
        entry_rows = cur.execute(
            """
            SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
            FROM decisions
            WHERE dry_run = 1
              AND action IN ('TRADE_YES','TRADE_NO')
              AND resolved_ts IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max_to_check),),
        ).fetchall()

        for r in entry_rows:
            checked += 1

            r_ts = _parse_iso(str(r["ts"] or ""))
            if r_ts is None:
                skipped += 1
                continue

            age_min = (now - r_ts).total_seconds() / 60.0
            if age_min < float(min_age_minutes):
                skipped += 1
                continue

            ticker = str(r["ticker"] or "").strip()
            if not ticker:
                skipped += 1
                continue

            action = str(r["action"] or "")
            reasons = str(r["reasons"] or "")

            entry_px = _extract_entry_price_cents(reasons)
            count = _extract_order_count(reasons)

            if entry_px is None or count is None or int(count) <= 0:
                skipped += 1
                continue

            mark_yes_i: Optional[int] = None
            mark_src = "unattempted"
            spread_i: Optional[int] = None
            vol_f: Optional[float] = None
            status_s: Optional[str] = None
            mtm_reason = "ok"

            yes_bid_i: Optional[int] = None
            yes_ask_i: Optional[int] = None

            # Fetch mark
            try:
                raw_market = client.get_market(str(ticker))
                market_obj = raw_market
                if isinstance(raw_market, dict) and isinstance(raw_market.get("market"), dict):
                    market_obj = raw_market["market"]

                s = summarize_market(market_obj)
                status_s = str(s.get("status", "") or "")

                try:
                    spread_i = None if s.get("spread_cents") is None else int(round(float(s.get("spread_cents"))))
                except Exception:
                    spread_i = None

                try:
                    vol_f = None if s.get("volume") is None else float(s.get("volume"))
                except Exception:
                    vol_f = None

                try:
                    yes_bid_i = None if s.get("yes_bid") is None else int(round(float(s.get("yes_bid"))))
                except Exception:
                    yes_bid_i = None

                try:
                    yes_ask_i = None if s.get("yes_ask") is None else int(round(float(s.get("yes_ask"))))
                except Exception:
                    yes_ask_i = None

                mark_yes_i, mark_src = _safe_mark_yes_from_summary(s)
                if mark_yes_i is None:
                    raise RuntimeError("no_usable_mark_yes")

                # Mark validity gate: reject degenerate/untradable marks
                unusable = False

                # If we have a spread, require it reasonably tight
                if spread_i is not None and int(spread_i) >= 30:
                    unusable = True

                # Classic degenerate book: 0/100
                if yes_bid_i == 0 and yes_ask_i == 100:
                    unusable = True

                # If your summarize_market mid is 50 while spread is 100, it's degenerate
                if mark_yes_i == 50 and spread_i == 100:
                    unusable = True

                # Status gate: only reject if we are sure it's not tradable
                # (keep this permissive to avoid false unusable)
                if status_s and status_s.strip():
                    st = status_s.strip().lower()
                    if st in ("closed", "settled", "finalized", "resolved", "cancelled", "canceled"):
                        unusable = True

                if unusable:
                    mtm_reason = "unusable_quote"
                    mark_src = f"rejected:{mark_src}"
                    mark_yes_i = None

            except Exception as e:
                mtm_reason = f"market_fetch_failed:{type(e).__name__}"
                mark_src = "fallback_zero"
                mark_yes_i = None

            # Delay policy for unusable quotes (do NOT resolve immediately)
            if mtm_reason == "unusable_quote":
                if age_min < float(unusable_defer_minutes):
                    skipped += 1
                    if logger:
                        logger.info(
                            "PaperResolver (MTM) defer unusable_quote "
                            f"id={int(r['id'])} age_min={age_min:.1f} ticker={ticker} "
                            f"status={status_s} spread={spread_i} bid={yes_bid_i} ask={yes_ask_i} mark_src={mark_src}"
                        )
                    continue

                if age_min < float(unusable_force_resolve_minutes):
                    skipped += 1
                    if logger:
                        logger.info(
                            "PaperResolver (MTM) keep-defer unusable_quote "
                            f"id={int(r['id'])} age_min={age_min:.1f} ticker={ticker} "
                            f"status={status_s} spread={spread_i} bid={yes_bid_i} ask={yes_ask_i} mark_src={mark_src}"
                        )
                    continue
                # else: fall through and force-resolve with pnl=0 + mtm_label:unusable_quote

            # Compute MTM pnl
            if mark_yes_i is None:
                pnl: float = 0.0
            else:
                pnl = _compute_mtm_pnl_from_mark(
                    action=action,
                    entry_px_side_cents=int(entry_px),
                    mark_yes_cents=int(mark_yes_i),
                    count=int(count),
                )

            try:
                pnl = float(pnl)
            except Exception:
                pnl = 0.0

            # Determine won flag (None if basically flat)
            if abs(pnl) < float(flat_pnl_epsilon):
                won_val_db: Optional[int] = None
                won_val_log: Optional[bool] = None
            else:
                won_val_log = bool(pnl > 0)
                won_val_db = 1 if won_val_log else 0

            # Tag for learning hygiene
            if mtm_reason == "ok":
                reasons2 = _append_tag(reasons, "mtm_label:good")
            elif mtm_reason == "unusable_quote":
                reasons2 = _append_tag(reasons, "mtm_label:unusable_quote")
            else:
                reasons2 = _append_tag(reasons, "mtm_label:market_fetch_failed")

            cur.execute(
                """
                UPDATE decisions
                SET realized_pnl = ?,
                    won = ?,
                    resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                """,
                (float(pnl), won_val_db, now_iso, reasons2, int(r["id"])),
            )

            updated += 1

            if logger:
                mark_no = int(max(1, min(99, 100 - int(mark_yes_i)))) if isinstance(mark_yes_i, int) else None
                side = "YES" if action == "TRADE_YES" else ("NO" if action == "TRADE_NO" else "?")
                logger.info(
                    "PaperResolver (MTM) resolved "
                    f"id={int(r['id'])} age_min={age_min:.1f} ticker={ticker} side={side} action={action} "
                    f"entry_px_side={int(entry_px)} count={int(count)} "
                    f"mark_yes={mark_yes_i} mark_no={mark_no} mark_src={mark_src} "
                    f"status={status_s} spread={spread_i} bid={yes_bid_i} ask={yes_ask_i} vol={vol_f} "
                    f"pnl={pnl:.4f} won={won_val_log} mtm_reason={mtm_reason}"
                )

        con.commit()

    except Exception:
        try:
            con.rollback()
        except Exception:
            pass
        raise
    finally:
        con.close()

    return PaperResolutionResult(updated=updated, checked=checked, skipped=skipped)