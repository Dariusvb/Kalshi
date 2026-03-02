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


# Bump this any time you change resolver logic materially.
_RESOLVER_VERSION = 4

# Basic number matcher for tolerant parsing
_RE_NUM = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def parse_kv_tokens(reasons: str) -> dict[str, str]:
    """
    Parse tolerant "k:v" tokens separated by semicolons into a dict (last-write-wins).

    Properties:
      - Ignores empty tokens
      - Trims whitespace
      - Accepts missing trailing ';'
      - Accepts duplicate keys (last wins)
      - Only splits on the first ':' so values may contain ':' later
      - Safe on None/bad inputs
    """
    out: dict[str, str] = {}
    if not reasons:
        return out

    s = str(reasons)
    for raw_tok in s.split(";"):
        tok = raw_tok.strip()
        if not tok or ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        out[k] = v
    return out


def _parse_iso(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        s = str(ts).strip().replace("Z", "+00:00")
        d = dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.UTC)
        return d.astimezone(dt.UTC)
    except Exception:
        return None


def _get_db_path_from_store(store: Any) -> str:
    """
    Best-effort DB path discovery.

    Hard requirement:
      check attrs in order: ("DB_PATH","db_path","path","_db_path")
      else default to "bot_state.db"
    """
    for attr in ("DB_PATH", "db_path", "path", "_db_path"):
        try:
            p = getattr(store, attr, None)
        except Exception:
            p = None
        if isinstance(p, str) and p.strip():
            return p.strip()
    return "bot_state.db"


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        return int(round(float(v)))
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        try:
            m = _RE_NUM.search(str(v))
            return float(m.group(1)) if m else None
        except Exception:
            return None


def _clamp_cents(v: Any) -> Optional[int]:
    """Accepts 1..99 inclusive. Returns None if out of range/unparseable."""
    i = _safe_int(v)
    if i is None:
        return None
    if 1 <= i <= 99:
        return i
    return None


def _append_tag(reasons: str, tag: str) -> str:
    rs = str(reasons or "")
    if not tag:
        return rs
    if tag in rs:
        return rs
    if rs and not rs.endswith(";"):
        rs += ";"
    return rs + tag + ";"


def _extract_entry_price_cents_from_tokens(toks: dict[str, str], reasons: str) -> Optional[int]:
    v = toks.get("entry_price_cents")
    if v is None:
        m = re.search(r"\bentry_price_cents\s*:\s*([-+]?\d+)", str(reasons or ""))
        v = m.group(1) if m else None
    return _clamp_cents(v)


def _extract_exit_price_cents_from_tokens(toks: dict[str, str], reasons: str) -> Optional[int]:
    v = toks.get("exit_price_cents")
    if v is None:
        v = toks.get("entry_price_cents")  # tolerant fallback
    if v is None:
        m = re.search(r"\bexit_price_cents\s*:\s*([-+]?\d+)", str(reasons or ""))
        v = m.group(1) if m else None
    return _clamp_cents(v)


def _extract_order_count_from_tokens(toks: dict[str, str], reasons: str) -> Optional[int]:
    v = toks.get("order_count")
    if v is None:
        m = re.search(r"\border_count\s*:\s*([-+]?\d+)", str(reasons or ""))
        v = m.group(1) if m else None
    i = _safe_int(v)
    if i is None:
        return None
    return int(max(0, i))


def _extract_exit_count_from_tokens(toks: dict[str, str], reasons: str) -> Optional[int]:
    v = toks.get("exit_count")
    if v is None:
        m = re.search(r"\bexit_count\s*:\s*([-+]?\d+)", str(reasons or ""))
        v = m.group(1) if m else None
    i = _safe_int(v)
    if i is None:
        return None
    return int(max(0, i))


def _extract_exit_entry_id_from_tokens(toks: dict[str, str], reasons: str) -> Optional[int]:
    v = toks.get("entry_id")
    if v is None:
        m = re.search(r"\bentry_id\s*:\s*([-+]?\d+)", str(reasons or ""))
        v = m.group(1) if m else None
    i = _safe_int(v)
    if i is None or i <= 0:
        return None
    return i


def _extract_yes_no_from_tokens(toks: dict[str, str], *, mode: str) -> Optional[str]:
    """
    Entry rows: order_yes_no:<yes|no>
    Exit rows:  exit_yes_no:<yes|no>
    """
    key = "order_yes_no" if mode == "entry" else "exit_yes_no"
    v = toks.get(key)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("yes", "no"):
        return s
    return None


def _infer_count_from_stake_if_present(toks: dict[str, str], entry_px_cents: int) -> Optional[int]:
    """
    If order_count is missing, try to infer from stake_dollars if present.

    shares ~= stake_dollars / (entry_px_cents/100.0)
    Conservative + clamped; last resort.
    """
    stake_raw = toks.get("stake_dollars")
    if stake_raw is None:
        return None
    stake = _safe_float(stake_raw)
    if stake is None or stake <= 0:
        return None

    px = int(max(1, min(99, int(entry_px_cents))))
    price = px / 100.0
    if price <= 0:
        return None

    est = int(stake / price + 0.5)
    if est <= 0:
        return None
    return int(min(est, 10_000))


def _compute_pnl_from_prices(*, entry_px: int, exit_px: int, count: int) -> float:
    """
    Realized/MTM PnL in *side price space*:
      pnl = (exit - entry) * count / 100.0

    NOTE: This is a naive paper PnL model (no fees/slippage/partials unless logged).
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
    MTM pnl using mark derived from YES quote space:
      - TRADE_YES: mark_px_side = mark_yes
      - TRADE_NO:  mark_px_side = 100 - mark_yes
    """
    a = str(action or "")
    c = int(max(0, int(count)))
    if c <= 0:
        return 0.0

    mark_yes_i = int(max(1, min(99, int(mark_yes_cents))))
    if a == "TRADE_YES":
        return _compute_pnl_from_prices(entry_px=entry_px_side_cents, exit_px=mark_yes_i, count=c)
    if a == "TRADE_NO":
        mark_no = int(max(1, min(99, 100 - mark_yes_i)))
        return _compute_pnl_from_prices(entry_px=entry_px_side_cents, exit_px=mark_no, count=c)
    return 0.0


def _unwrap_market_payload(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        for outer in ("market", "data", "result", "response"):
            v = raw.get(outer)
            if isinstance(v, dict):
                if "ticker" in v:
                    return v
                for inner in ("market", "data", "result", "response"):
                    vv = v.get(inner)
                    if isinstance(vv, dict) and "ticker" in vv:
                        return vv
        if "ticker" in raw:
            return raw
    return None


def _client_fetch_market_summary(
    *,
    client: Any,
    ticker: str,
    logger: Any = None,
    fallback_scan_pages: int = 2,
    fallback_scan_limit: int = 200,
) -> tuple[Optional[dict], str]:
    """
    Bulletproof market fetch:
      1) try client.get_market(ticker)
      2) fallback: client.get_markets(limit=..., cursor=...) and scan for ticker

    Returns (summary_dict, fetch_mode)
    """
    t = str(ticker or "").strip()
    if not t:
        return None, "no_ticker"

    try:
        gm = getattr(client, "get_market", None)
        if callable(gm):
            raw = gm(t)
            mo = _unwrap_market_payload(raw) if raw is not None else None
            if mo:
                return summarize_market(mo), "get_market"
    except Exception as e:
        if logger:
            logger.warning(f"PaperResolver: get_market failed for {t}: {type(e).__name__}:{e}")

    try:
        gms = getattr(client, "get_markets", None)
        if not callable(gms):
            return None, "no_get_markets"

        cursor: Optional[str] = None
        for _ in range(max(1, int(fallback_scan_pages))):
            try:
                raw = gms(limit=int(fallback_scan_limit), cursor=cursor)
            except TypeError:
                raw = gms(limit=int(fallback_scan_limit))

            markets = None
            if isinstance(raw, dict):
                if isinstance(raw.get("markets"), list):
                    markets = raw.get("markets")
                else:
                    for k in ("data", "result", "response"):
                        v = raw.get(k)
                        if isinstance(v, dict) and isinstance(v.get("markets"), list):
                            markets = v.get("markets")
                            break
                        if isinstance(v, list):
                            markets = v
                            break
            elif isinstance(raw, list):
                markets = raw

            if not isinstance(markets, list):
                break

            for m in markets:
                if isinstance(m, dict) and str(m.get("ticker") or "").strip() == t:
                    try:
                        return summarize_market(m), "scan_get_markets"
                    except Exception:
                        return None, "scan_get_markets_summarize_failed"

            if isinstance(raw, dict):
                cursor = raw.get("cursor")
                if not cursor:
                    break
            else:
                break

    except Exception as e:
        if logger:
            logger.warning(f"PaperResolver: scan_get_markets failed for {t}: {type(e).__name__}:{e}")
        return None, "scan_get_markets_failed"

    return None, "not_found"


def _validate_entry_link(
    *,
    entry_row: sqlite3.Row,
    entry_action_expected: str,
    ticker_expected: str,
    exit_ts: dt.datetime,
    exit_side_yes_no: Optional[str],
    exit_cnt: Optional[int],
) -> tuple[bool, str]:
    """
    Validates an entry row as a candidate for an exit link.

    We treat entry_id as a hint, not truth. This prevents BAD_LINK-style mislabels when
    the bot writes wrong entry_id tags.

    Returns (ok, reason_string).
    """
    try:
        if entry_row is None:
            return False, "missing_entry_row"

        if entry_row["resolved_ts"] is not None:
            return False, "entry_already_resolved"

        tkr = str(entry_row["ticker"] or "").strip()
        if tkr != str(ticker_expected or "").strip():
            return False, "ticker_mismatch"

        act = str(entry_row["action"] or "")
        if act != str(entry_action_expected or ""):
            return False, "action_mismatch"

        ent_ts = _parse_iso(str(entry_row["ts"] or ""))
        if ent_ts is None:
            return False, "bad_entry_ts"

        # Entry should be <= exit time (allow small skew)
        skew = dt.timedelta(minutes=2)
        if ent_ts > (exit_ts + skew):
            return False, "entry_after_exit"

        # Optional: yes/no token consistency
        en_toks = parse_kv_tokens(str(entry_row["reasons"] or ""))
        entry_yes_no = _extract_yes_no_from_tokens(en_toks, mode="entry")
        if exit_side_yes_no is not None and entry_yes_no is not None and exit_side_yes_no != entry_yes_no:
            return False, "yesno_token_mismatch"

        # Optional: count compatibility
        entry_px = _extract_entry_price_cents_from_tokens(en_toks, str(entry_row["reasons"] or ""))
        if entry_px is None:
            return False, "missing_entry_px"

        entry_cnt = _extract_order_count_from_tokens(en_toks, str(entry_row["reasons"] or ""))
        if entry_cnt is None or int(entry_cnt) <= 0:
            # allow missing count; resolver can infer/default later
            entry_cnt = None

        if exit_cnt is not None and entry_cnt is not None:
            if int(exit_cnt) > int(entry_cnt) and int(entry_cnt) > 0:
                # Not fatal, but suspicious; treat as invalid to avoid mismapping.
                return False, "exit_cnt_gt_entry_cnt"

        # Avoid reusing entry that already has exit_resolved tag (even if resolved_ts is null due to corruption)
        rs = str(entry_row["reasons"] or "")
        if "exit_resolved:1" in rs:
            return False, "entry_already_tagged_resolved"

        return True, "ok"
    except Exception:
        return False, "exception_in_validate"


def _score_entry_candidate(
    *,
    entry_row: sqlite3.Row,
    exit_ts: dt.datetime,
    exit_yes_no: Optional[str],
    exit_cnt: Optional[int],
) -> tuple[int, str]:
    """
    Heuristic scoring to pick the most plausible entry when entry_id is missing/bad.

    Higher is better. Returns (score, explanation).
    """
    score = 0
    why: list[str] = []

    ent_ts = _parse_iso(str(entry_row["ts"] or ""))
    if ent_ts is None:
        return -10_000, "bad_entry_ts"

    # Prefer entries at/before exit time; heavily penalize after-exit entries
    delta_min = (exit_ts - ent_ts).total_seconds() / 60.0
    if delta_min >= -2.0:  # allow up to 2 min clock skew
        score += 20
        why.append("ts_ok")
    else:
        score -= 10_000
        why.append("ts_after_exit")

    # Closer in time is better (cap influence)
    # e.g. delta=0 -> +20; delta=60 -> +0; delta>60 -> +0
    if delta_min >= 0:
        score += int(max(0.0, 20.0 - min(60.0, delta_min) * (20.0 / 60.0)))
        why.append(f"ts_closeness:{delta_min:.1f}m")

    rs = str(entry_row["reasons"] or "")
    toks = parse_kv_tokens(rs)

    # Token yes/no match, if available
    entry_yes_no = _extract_yes_no_from_tokens(toks, mode="entry")
    if exit_yes_no is not None and entry_yes_no is not None:
        if exit_yes_no == entry_yes_no:
            score += 30
            why.append("yesno_match")
        else:
            score -= 200
            why.append("yesno_mismatch")

    # Count compatibility
    entry_cnt = _extract_order_count_from_tokens(toks, rs)
    if exit_cnt is not None and entry_cnt is not None and int(entry_cnt) > 0:
        if int(exit_cnt) <= int(entry_cnt):
            score += 10
            why.append("cnt_ok")
        else:
            score -= 50
            why.append("cnt_bad")

    # Require entry_price_cents to exist
    entry_px = _extract_entry_price_cents_from_tokens(toks, rs)
    if entry_px is not None:
        score += 5
        why.append("px_ok")
    else:
        score -= 200
        why.append("px_missing")

    # Penalize already-tagged-resolved rows (corrupt state guard)
    if "exit_resolved:1" in rs:
        score -= 1000
        why.append("tagged_resolved")

    return score, ",".join(why)


def _link_confidence_from_score(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "med"
    return "low"


def _select_fillable_mtm_mark_yes(
    *,
    action: str,
    summary: dict,
    mid_spread_gate_cents: int = 10,
) -> tuple[Optional[int], str, str]:
    """
    Choose a more "fillable" MTM mark to reduce bias in thin markets.

    Policy:
      - For TRADE_YES (closing a YES position): prefer YES bid (conservative liquidation)
      - For TRADE_NO  (closing a NO position): prefer NO bid ~= 100 - YES ask
        (equivalently, pass mark_yes = YES ask so mark_no = 100 - mark_yes)

      - Only allow mid/avg_bid_ask when spread is reasonably tight (<= gate)
      - If only one-sided quote is available in a way that isn't conservative, treat as unusable_quote.

    Returns:
      (mark_yes_cents_or_None, mark_src, mtm_conf)
    """
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

    a = str(action or "")
    yes_bid = _to_int(summary.get("yes_bid"))
    yes_ask = _to_int(summary.get("yes_ask"))
    mid_yes = _to_int(summary.get("mid_yes_cents"))

    spread = None
    try:
        if summary.get("spread_cents") is not None:
            spread = int(round(float(summary.get("spread_cents"))))
    except Exception:
        spread = None

    # Confidence is about quote quality, not about market direction.
    # "high" means two-sided and relatively tight.
    mtm_conf = "low"
    if yes_bid is not None and yes_ask is not None:
        if spread is not None and spread <= int(mid_spread_gate_cents):
            mtm_conf = "high"
        else:
            mtm_conf = "med"  # two-sided but wide

    # Conservative fillable mark
    if a == "TRADE_YES":
        if yes_bid is not None:
            return yes_bid, "fillable_yes_bid", mtm_conf
        # If no bid, we cannot conservatively liquidate; avoid using ask-only as it biases optimistic.
        # Allow mid only if tight spread and both quotes exist (but that implies bid exists anyway).
        if yes_bid is None and yes_ask is None and mid_yes is not None:
            return None, "no_quote", "low"
        return None, "no_yes_bid", "low"

    if a == "TRADE_NO":
        # For NO liquidation, conservative mark uses NO bid ~= 100 - YES ask -> pass mark_yes = YES ask.
        if yes_ask is not None:
            return yes_ask, "fillable_no_bid_via_yes_ask", mtm_conf
        return None, "no_yes_ask", "low"

    # Unknown action => unusable
    return None, "bad_action", "low"


def resolve_paper_trades(
    *,
    store: Any,
    client: Any,
    logger: Any = None,
    max_to_check: int = 30,
    min_age_minutes: int = 10,
) -> PaperResolutionResult:
    """
    DRY_RUN trade lifecycle resolver with:
      - safer exit->entry mapping (entry_id is hint; scored fallback)
      - more fillable MTM marks (reduce mid/one-sided bias)
      - richer audit tags (link_mode/link_confidence/mtm_conf/pnl_model)

    Hard requirements:
      - Exports PaperResolutionResult, PaperResolveResult alias, and resolve_paper_trades()
      - Uses sqlite3 directly and discovers DB path from store attrs
      - Reads/updates decisions table; tolerant token parsing; never crashes on bad data

    See inline comments for mapping/mark policies.
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

    # Basic table existence check
    try:
        cur.execute("SELECT 1 FROM decisions LIMIT 1;").fetchone()
    except Exception as e:
        con.close()
        raise RuntimeError(f"decisions table not found or unreadable in {db_path}: {e}")

    # MTM policy knobs
    mtm_fallback_minutes = max(float(min_age_minutes), 60.0)
    unusable_force_resolve_minutes = 12 * 60.0  # 12 hours

    # Exit-entry fallback scan depth (small N is enough and keeps it fast)
    fallback_entry_scan_limit = 25

    try:
        try:
            cur.execute("BEGIN;")
        except Exception:
            pass

        # =========================================================
        # PASS A: explicit exits (primary path)
        # =========================================================
        exit_rows = cur.execute(
            """
            SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
            FROM decisions
            WHERE dry_run = 1
              AND resolved_ts IS NULL
              AND action IN ('TRADE_EXIT_YES', 'TRADE_EXIT_NO')
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max_to_check),),
        ).fetchall()

        for ex in exit_rows:
            checked += 1

            ex_id = int(ex["id"])
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
            entry_action_expected = "TRADE_YES" if held_side == "YES" else "TRADE_NO"

            exit_reasons = str(ex["reasons"] or "")
            ex_toks = parse_kv_tokens(exit_reasons)

            exit_px = _extract_exit_price_cents_from_tokens(ex_toks, exit_reasons)
            exit_cnt = _extract_exit_count_from_tokens(ex_toks, exit_reasons)
            prefer_entry_id = _extract_exit_entry_id_from_tokens(ex_toks, exit_reasons)
            exit_yes_no = _extract_yes_no_from_tokens(ex_toks, mode="exit")  # optional

            if exit_px is None:
                skipped += 1
                continue

            ent: Optional[sqlite3.Row] = None
            link_mode = "unknown"
            link_conf = "low"
            link_note = ""

            # -----------------------------
            # 1) Try entry_id pointer (HINT)
            # -----------------------------
            if prefer_entry_id is not None:
                pointed = cur.execute(
                    """
                    SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
                    FROM decisions
                    WHERE dry_run = 1
                      AND id = ?
                      AND action IN ('TRADE_YES','TRADE_NO')
                    """,
                    (int(prefer_entry_id),),
                ).fetchone()

                if pointed is not None:
                    ok, reason = _validate_entry_link(
                        entry_row=pointed,
                        entry_action_expected=entry_action_expected,
                        ticker_expected=ticker,
                        exit_ts=ex_ts,
                        exit_side_yes_no=exit_yes_no,
                        exit_cnt=exit_cnt,
                    )
                    if ok:
                        ent = pointed
                        link_mode = "entry_id"
                        link_conf = "high"
                        link_note = reason
                    else:
                        # Entry-id is bad; do NOT crash, just warn and fall back.
                        if logger:
                            logger.warning(
                                "PaperResolver BAD_LINK override entry_id hint "
                                f"exit_id={ex_id} hinted_entry_id={prefer_entry_id} reason={reason}"
                            )
                        ent = None

            # ---------------------------------------------
            # 2) Scored fallback across recent candidates
            # ---------------------------------------------
            if ent is None:
                candidates = cur.execute(
                    """
                    SELECT id, ts, ticker, action, dry_run, reasons, resolved_ts
                    FROM decisions
                    WHERE dry_run = 1
                      AND ticker = ?
                      AND action = ?
                      AND resolved_ts IS NULL
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (ticker, entry_action_expected, int(fallback_entry_scan_limit)),
                ).fetchall()

                best: Optional[sqlite3.Row] = None
                best_score = -10_000
                best_why = ""

                for cand in candidates:
                    ok, _ = _validate_entry_link(
                        entry_row=cand,
                        entry_action_expected=entry_action_expected,
                        ticker_expected=ticker,
                        exit_ts=ex_ts,
                        exit_side_yes_no=exit_yes_no,
                        exit_cnt=exit_cnt,
                    )
                    if not ok:
                        continue

                    sc, why = _score_entry_candidate(
                        entry_row=cand,
                        exit_ts=ex_ts,
                        exit_yes_no=exit_yes_no,
                        exit_cnt=exit_cnt,
                    )
                    if sc > best_score:
                        best_score = sc
                        best = cand
                        best_why = why

                if best is None:
                    skipped += 1
                    continue

                ent = best
                link_mode = "fallback_scored"
                link_conf = _link_confidence_from_score(int(best_score))
                link_note = f"score={best_score};why={best_why}"

                # If exit had a hinted entry_id but we didn't use it, note it.
                if prefer_entry_id is not None and logger:
                    logger.warning(
                        "PaperResolver used fallback_scored over hinted entry_id "
                        f"exit_id={ex_id} hinted_entry_id={prefer_entry_id} chosen_entry_id={int(ent['id'])} "
                        f"{link_note}"
                    )

            # If linked entry somehow resolved between selection and update, skip safely
            if ent is None or ent["resolved_ts"] is not None:
                skipped += 1
                continue

            entry_id = int(ent["id"])
            entry_reasons = str(ent["reasons"] or "")
            en_toks = parse_kv_tokens(entry_reasons)

            entry_px = _extract_entry_price_cents_from_tokens(en_toks, entry_reasons)
            entry_cnt = _extract_order_count_from_tokens(en_toks, entry_reasons)

            if entry_px is None:
                skipped += 1
                continue

            if entry_cnt is None or int(entry_cnt) <= 0:
                inferred = _infer_count_from_stake_if_present(en_toks, int(entry_px))
                entry_cnt = inferred if inferred is not None else 1

            # Use min(entry_cnt, exit_cnt) if exit_count provided
            count_used = int(max(0, int(entry_cnt)))
            if exit_cnt is not None and int(exit_cnt) > 0:
                count_used = int(min(count_used, int(exit_cnt)))

            if count_used <= 0:
                skipped += 1
                continue

            pnl = _compute_pnl_from_prices(entry_px=int(entry_px), exit_px=int(exit_px), count=count_used)

            # won = 1 if pnl > 0 else 0 if pnl < 0 else None if abs(pnl) < 0.0001.
            if abs(float(pnl)) < 0.0001:
                won_db: Optional[int] = None
            else:
                won_db = 1 if float(pnl) > 0 else 0

            # Tags (entry)
            entry_new_reasons = entry_reasons
            for tg in (
                "exit_resolved:1",
                f"exit_id:{ex_id}",
                f"exit_px_cents:{int(exit_px)}",
                f"exit_cnt:{int(count_used)}",
                f"link_mode:{link_mode}",
                f"link_confidence:{link_conf}",
                "pnl_model:naive",
                f"resolver_version:{_RESOLVER_VERSION}",
            ):
                entry_new_reasons = _append_tag(entry_new_reasons, tg)

            # Helpful debug-only note (kept short; last-write-wins dict parser won't care)
            if link_mode == "fallback_scored":
                # include a compact hint, avoid exploding reasons length
                entry_new_reasons = _append_tag(entry_new_reasons, "link_note:fallback_scored")

            # Tags (exit)
            exit_new_reasons = exit_reasons
            for tg in (
                "exit_consumed:1",
                f"entry_id:{entry_id}",
                "exit_resolve_reason:explicit",
                f"link_mode:{link_mode}",
                f"link_confidence:{link_conf}",
                f"resolver_version:{_RESOLVER_VERSION}",
            ):
                exit_new_reasons = _append_tag(exit_new_reasons, tg)

            # Guarded entry update: only resolve if still unresolved
            cur.execute(
                """
                UPDATE decisions
                SET realized_pnl = ?,
                    won = ?,
                    resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                  AND resolved_ts IS NULL
                """,
                (float(pnl), won_db, now_iso, entry_new_reasons, entry_id),
            )
            entry_rc = int(getattr(cur, "rowcount", 0) or 0)
            if entry_rc != 1:
                skipped += 1
                if logger:
                    logger.warning(
                        "PaperResolver (EXIT) guarded-skip (entry not updated) "
                        f"entry_id={entry_id} exit_id={ex_id} rowcount={entry_rc}"
                    )
                continue

            # Guarded exit consumption
            cur.execute(
                """
                UPDATE decisions
                SET resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                  AND resolved_ts IS NULL
                """,
                (now_iso, exit_new_reasons, ex_id),
            )

            updated += 1
            if logger:
                logger.info(
                    "PaperResolver (EXIT) resolved "
                    f"ticker={ticker} side={held_side} entry_id={entry_id} exit_id={ex_id} "
                    f"link_mode={link_mode} link_conf={link_conf} "
                    f"entry_px={int(entry_px)} exit_px={int(exit_px)} count={count_used} "
                    f"pnl={float(pnl):.4f} won={won_db}"
                )

        # =========================================================
        # PASS B: MTM fallback for stale entries (secondary path)
        # =========================================================
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

            r_id = int(r["id"])
            r_ts = _parse_iso(str(r["ts"] or ""))
            if r_ts is None:
                skipped += 1
                continue

            age_min = (now - r_ts).total_seconds() / 60.0
            if age_min < float(mtm_fallback_minutes):
                skipped += 1
                continue

            ticker = str(r["ticker"] or "").strip()
            if not ticker:
                skipped += 1
                continue

            action = str(r["action"] or "")
            reasons = str(r["reasons"] or "")
            toks = parse_kv_tokens(reasons)

            entry_px = _extract_entry_price_cents_from_tokens(toks, reasons)
            count = _extract_order_count_from_tokens(toks, reasons)

            if entry_px is None:
                skipped += 1
                continue

            if count is None or int(count) <= 0:
                inferred = _infer_count_from_stake_if_present(toks, int(entry_px))
                count = inferred if inferred is not None else 1

            mtm_label = "good"
            mtm_conf = "low"
            mark_yes: Optional[int] = None
            mark_src = "unattempted"

            try:
                summary, fetch_mode = _client_fetch_market_summary(
                    client=client,
                    ticker=ticker,
                    logger=logger,
                    fallback_scan_pages=2,
                    fallback_scan_limit=200,
                )
                if summary is None:
                    raise RuntimeError(f"market_not_found:{fetch_mode}")

                # Choose a more fillable mark to reduce systematic bias
                mark_yes, src, conf = _select_fillable_mtm_mark_yes(action=action, summary=summary)
                mtm_conf = conf
                mark_src = f"{fetch_mode}:{src}"

                if mark_yes is None:
                    mtm_label = "unusable_quote"

            except Exception as e:
                mtm_label = "market_fetch_failed"
                mtm_conf = "low"
                mark_src = f"failed:{type(e).__name__}"
                mark_yes = None

            # Unusable quote policy:
            #   - Defer until age >= 12 hours
            #   - Then force resolve pnl=0.0 and won=None (prevents never-ending unresolved rows)
            if mtm_label == "unusable_quote" and age_min < float(unusable_force_resolve_minutes):
                skipped += 1
                if logger:
                    logger.info(
                        "PaperResolver (MTM) defer unusable_quote "
                        f"id={r_id} age_min={age_min:.1f} ticker={ticker} action={action} "
                        f"mtm_conf={mtm_conf} mark_src={mark_src}"
                    )
                continue

            # Compute MTM pnl (naive model; conservative mark reduces optimism)
            if mark_yes is None:
                pnl = 0.0
            else:
                pnl = _compute_mtm_pnl_from_mark(
                    action=action,
                    entry_px_side_cents=int(entry_px),
                    mark_yes_cents=int(mark_yes),
                    count=int(count),
                )

            # won for MTM:
            #   if abs(pnl) < 0.01 => won=None else 1/0
            if abs(float(pnl)) < 0.01:
                won_db = None
            else:
                won_db = 1 if float(pnl) > 0 else 0

            # Tag reasons
            reasons2 = reasons
            for tg in (
                f"resolver_version:{_RESOLVER_VERSION}",
                f"mtm_label:{mtm_label}",
                f"mtm_conf:{mtm_conf}",
                "pnl_model:naive",
            ):
                reasons2 = _append_tag(reasons2, tg)

            # Guarded update
            cur.execute(
                """
                UPDATE decisions
                SET realized_pnl = ?,
                    won = ?,
                    resolved_ts = ?,
                    reasons = ?
                WHERE id = ?
                  AND resolved_ts IS NULL
                """,
                (float(pnl), won_db, now_iso, reasons2, r_id),
            )
            rc = int(getattr(cur, "rowcount", 0) or 0)
            if rc != 1:
                skipped += 1
                if logger:
                    logger.warning(
                        "PaperResolver (MTM) guarded-skip (row not updated) "
                        f"id={r_id} rowcount={rc} ticker={ticker}"
                    )
                continue

            updated += 1

            if logger:
                side = "YES" if action == "TRADE_YES" else ("NO" if action == "TRADE_NO" else "?")
                mark_no = (100 - int(mark_yes)) if isinstance(mark_yes, int) else None
                logger.info(
                    "PaperResolver (MTM) resolved "
                    f"id={r_id} age_min={age_min:.1f} ticker={ticker} side={side} "
                    f"entry_px_side={int(entry_px)} count={int(count)} "
                    f"mark_yes={mark_yes} mark_no={mark_no} mark_src={mark_src} "
                    f"pnl={float(pnl):.4f} won={won_db} mtm_label={mtm_label} mtm_conf={mtm_conf}"
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


# -----------------------------
# Optional self-checks (won't run unless you execute this module directly)
# -----------------------------
def run_self_checks() -> None:
    # parse_kv_tokens last-write-wins + whitespace tolerance
    rs = "foo; entry_price_cents: 51 ;order_count:2;order_count:3; bar"
    toks = parse_kv_tokens(rs)
    assert toks["entry_price_cents"] == "51"
    assert toks["order_count"] == "3"

    # exit parsing
    rs2 = "exit_price_cents:49; exit_count: 2 ; entry_id:123;"
    toks2 = parse_kv_tokens(rs2)
    assert _extract_exit_price_cents_from_tokens(toks2, rs2) == 49
    assert _extract_exit_count_from_tokens(toks2, rs2) == 2
    assert _extract_exit_entry_id_from_tokens(toks2, rs2) == 123

    # clamp checks
    assert _clamp_cents(0) is None
    assert _clamp_cents(1) == 1
    assert _clamp_cents(99) == 99
    assert _clamp_cents(100) is None

    # pnl sanity checks
    assert _compute_pnl_from_prices(entry_px=50, exit_px=60, count=2) == 0.2
    # NO trade: mark_yes=40 => mark_no=60; entry_no=60 => pnl 0
    assert _compute_mtm_pnl_from_mark(action="TRADE_NO", entry_px_side_cents=60, mark_yes_cents=40, count=1) == 0.0

    # MTM mark selection sanity
    summary = {"yes_bid": 40, "yes_ask": 60, "mid_yes_cents": 50, "spread_cents": 20}
    my, src, conf = _select_fillable_mtm_mark_yes(action="TRADE_YES", summary=summary)
    assert my == 40 and "yes_bid" in src
    my2, src2, conf2 = _select_fillable_mtm_mark_yes(action="TRADE_NO", summary=summary)
    assert my2 == 60 and "yes_ask" in src2

    print("paper_resolver.py self-checks passed.")


if __name__ == "__main__":
    run_self_checks()