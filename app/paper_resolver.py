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
    """
    if not reasons:
        return None
    s = str(reasons)
    idx = s.find(prefix)
    if idx < 0:
        return None
    tail = s[idx + len(prefix):]

    # Stop at ';' if present; otherwise take whole tail
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


def _extract_entry_price_cents(reasons: str) -> Optional[int]:
    px = _parse_int_after("entry_price_cents:", reasons or "")
    if px is None:
        return None
    if 1 <= px <= 99:
        return px
    return None


def _extract_exit_price_cents(reasons: str) -> Optional[int]:
    """
    Exit rows SHOULD log exit_price_cents, but if your exit code logs entry_price_cents:<exit_px>,
    we support both.
    """
    px = _parse_int_after("exit_price_cents:", reasons or "")
    if px is None:
        px = _parse_int_after("entry_price_cents:", reasons or "")
    if px is None:
        return None
    if 1 <= px <= 99:
        return px
    return None


def _extract_order_count(reasons: str) -> Optional[int]:
    c = _parse_int_after("order_count:", reasons or "")
    if c is None:
        return None
    return int(max(0, c))


def _extract_exit_count(reasons: str) -> Optional[int]:
    c = _parse_int_after("exit_count:", reasons or "")
    if c is None:
        return None
    return int(max(0, c))


def _compute_realized_pnl_from_prices(*, entry_px: int, exit_px: int, count: int) -> float:
    """
    PnL in *side price space* (YES prices for YES trades, NO prices for NO trades):
      pnl = (exit - entry) * count / 100
    """
    e = int(max(1, min(99, entry_px)))
    x = int(max(1, min(99, exit_px)))
    c = int(max(0, count))
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
    """
    a = str(action or "")
    c = int(max(0, count))
    if c <= 0:
        return 0.0

    mark_yes = int(max(1, min(99, mark_yes_cents)))

    if a == "TRADE_YES":
        entry_yes = int(max(1, min(99, entry_px_side_cents)))
        return _compute_realized_pnl_from_prices(entry_px=entry_yes, exit_px=mark_yes, count=c)

    if a == "TRADE_NO":
        mark_no = int(max(1, min(99, 100 - mark_yes)))
        entry_no = int(max(1, min(99, entry_px_side_cents)))
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

    Pass 2) MTM resolve old entries with no exits:
      - Uses current mid_yes from market
      - Uses entry_price_cents + order_count
      - NO trades use mark_no = 100 - mid_yes
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

        if entry_px is None or entry_cnt is None or entry_cnt <= 0:
            skipped += 1
            continue

        count_used = int(entry_cnt)
        if exit_cnt is not None and exit_cnt > 0:
            count_used = int(min(count_used, exit_cnt))
        if count_used <= 0:
            skipped += 1
            continue

        pnl = _compute_realized_pnl_from_prices(entry_px=int(entry_px), exit_px=int(exit_px), count=int(count_used))

        if pnl > 0:
            won_val: Optional[int] = 1
        elif pnl < 0:
            won_val = 0
        else:
            won_val = 0

        cur.execute(
            """
            UPDATE decisions
            SET realized_pnl = ?,
                won = ?,
                resolved_ts = ?
            WHERE id = ?
            """,
            (float(pnl), int(won_val), now_iso, entry_id),
        )

        cur.execute(
            """
            UPDATE decisions
            SET resolved_ts = ?
            WHERE id = ?
            """,
            (now_iso, int(ex["id"])),
        )

        con.commit()
        updated += 1

        if logger:
            logger.info(
                f"PaperResolver (EXIT) resolved ticker={ticker} side={held_side} "
                f"entry_id={entry_id} exit_id={int(ex['id'])} entry_px={entry_px} exit_px={exit_px} "
                f"count={count_used} pnl={pnl:.4f} won={won_val}"
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

    # Guardrails
    max_spread_cents = 12
    min_volume = 1.0
    flat_pnl_epsilon = 0.01  # don't label pennies as win/loss

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

        # Hard requirement: MTM needs entry px + count (no stake inference)
        if entry_px is None or count is None or count <= 0:
            skipped += 1
            continue

        # Pull current mark (mid_yes)
        try:
            raw_market = client.get_market(str(ticker))
            market_obj = raw_market
            if isinstance(raw_market, dict) and "market" in raw_market and isinstance(raw_market["market"], dict):
                market_obj = raw_market["market"]

            s = summarize_market(market_obj)

            spread = s.get("spread_cents", None)
            vol = s.get("volume", None)
            mid_yes = s.get("mid_yes_cents", None)

            spread_i = None
            try:
                spread_i = None if spread is None else int(round(float(spread)))
            except Exception:
                spread_i = None

            vol_f = None
            try:
                vol_f = None if vol is None else float(vol)
            except Exception:
                vol_f = None

            mid_i = None
            try:
                mid_i = None if mid_yes is None else int(round(float(mid_yes)))
            except Exception:
                mid_i = None

            if mid_i is None or not (1 <= mid_i <= 99):
                skipped += 1
                continue

            if spread_i is not None and spread_i > int(max_spread_cents):
                skipped += 1
                continue

            if vol_f is not None and vol_f < float(min_volume):
                skipped += 1
                continue

        except Exception as e:
            if logger:
                logger.warning(f"PaperResolver (MTM) market fetch failed for {ticker}: {e}")
            skipped += 1
            continue

        pnl = _compute_mtm_pnl_from_mark(
            action=action,
            entry_px_side_cents=int(entry_px),
            mark_yes_cents=int(mid_i),
            count=int(count),
        )

        if abs(pnl) < float(flat_pnl_epsilon):
            won_val = None
        else:
            won_val = bool(pnl > 0)

        cur.execute(
            """
            UPDATE decisions
            SET realized_pnl = ?,
                won = ?,
                resolved_ts = ?
            WHERE id = ?
            """,
            (
                float(pnl),
                None if won_val is None else (1 if won_val else 0),
                now_iso,
                int(r["id"]),
            ),
        )

        con.commit()
        updated += 1

        if logger:
            logger.info(
                f"PaperResolver (MTM) resolved id={int(r['id'])} ticker={ticker} action={action} "
                f"entry_px={entry_px} count={count} mark_yes={mid_i} pnl={pnl:.4f} won={won_val}"
            )

    con.close()
    return PaperResolutionResult(updated=updated, checked=checked, skipped=skipped)