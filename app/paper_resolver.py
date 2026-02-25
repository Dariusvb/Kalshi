from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Optional

from app.market_data import summarize_market


@dataclass
class PaperResolutionResult:
    updated: int
    checked: int
    skipped: int


def _parse_iso(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        ts = ts.replace("Z", "+00:00")  # tolerate Z suffix
        parsed = dt.datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(dt.UTC)
    except Exception:
        return None


def _extract_entry_price_cents_from_reasons(reasons: str) -> Optional[int]:
    """
    Optional parser if 'entry_price_cents:XX' is stored in reasons.
    """
    if not reasons:
        return None
    try:
        for part in str(reasons).split(";"):
            p = part.strip().lower()
            if p.startswith("entry_price_cents:"):
                return int(p.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def _estimate_entry_price_from_decision(row: dict) -> Optional[int]:
    """
    Fallback entry price estimate from conviction / direction when exact entry isn't logged.
    This is a proxy only. Better: store actual simulated entry price in decision row.
    """
    action = str(row.get("action", ""))
    direction = str(row.get("direction", "SKIP")).upper()
    if not action.startswith("TRADE_") or direction not in {"YES", "NO"}:
        return None

    conv = float(row.get("final_conviction", 50.0))
    implied = int(round(max(5, min(95, 50 + (conv - 50) * 0.35))))

    if direction == "YES":
        return implied

    # For NO buys, convert estimated NO entry into yes-space reference
    no_price = implied
    return 100 - no_price


def _paper_pnl_from_mark(
    *,
    direction: str,
    stake_dollars: float,
    entry_yes_price_cents: int,
    mark_yes_price_cents: int,
) -> float:
    """
    Approximate paper PnL using yes-price mark-to-market.
    Ignores fees/slippage/partial fills (MVP approximation).
    """
    direction = direction.upper()
    entry_yes = max(1, min(99, int(entry_yes_price_cents)))
    mark_yes = max(1, min(99, int(mark_yes_price_cents)))
    stake = max(0.0, float(stake_dollars))

    if stake <= 0:
        return 0.0

    if direction == "YES":
        entry_cost = entry_yes / 100.0
        contracts = stake / entry_cost if entry_cost > 0 else 0.0
        pnl = contracts * ((mark_yes - entry_yes) / 100.0)
        return round(pnl, 4)

    if direction == "NO":
        entry_no = (100 - entry_yes) / 100.0
        mark_no = (100 - mark_yes) / 100.0
        contracts = stake / entry_no if entry_no > 0 else 0.0
        pnl = contracts * (mark_no - entry_no)
        return round(pnl, 4)

    return 0.0


def _extract_minimal_market_quality(summary: dict) -> dict:
    """
    Pull a few quality indicators from summarize_market with safe defaults.
    This avoids relying too heavily on exact schema shape.
    """
    spread = summary.get("spread_cents", None)
    volume = summary.get("volume", None)
    mid_yes = summary.get("mid_yes_cents", None)
    status = str(summary.get("status", "unknown")).lower()

    try:
        spread = None if spread is None else int(round(float(spread)))
    except Exception:
        spread = None

    try:
        volume = None if volume is None else float(volume)
    except Exception:
        volume = None

    try:
        mid_yes = None if mid_yes is None else int(round(float(mid_yes)))
    except Exception:
        mid_yes = None

    return {
        "spread_cents": spread,
        "volume": volume,
        "mid_yes_cents": mid_yes,
        "status": status,
    }


def resolve_paper_trades(
    *,
    store: Any,
    client: Any,
    logger: Any = None,
    max_to_check: int = 25,
    min_age_minutes: int = 15,
) -> PaperResolutionResult:
    """
    Mark unresolved DRY_RUN trades as 'resolved' after they are old enough, using current market mark.
    This enables simulated W/L and PnL stats for learning + Discord summaries.

    Requirements:
    - store.unresolved_trade_decisions(...)
    - store.update_decision_outcome(...)
    - client.get_market(ticker)
    - market_data.summarize_market(...)
    """
    rows = store.unresolved_trade_decisions(limit=max_to_check, dry_run=True)
    now = dt.datetime.now(dt.UTC)

    updated = 0
    checked = 0
    skipped = 0

    # Guardrails so learning isn't trained on garbage marks.
    # These are intentionally permissive (not too conservative).
    max_resolve_spread_cents = 12
    min_resolve_volume = 1.0

    # If pnl is near-zero, don't force a win/loss label.
    flat_pnl_epsilon = 0.10  # $0.10

    for row in rows:
        checked += 1

        ts = _parse_iso(str(row.get("ts", "")))
        if ts is None:
            skipped += 1
            if logger:
                logger.debug(f"Paper resolver skip id={row.get('id')} reason=bad_ts")
            continue

        age_min = (now - ts).total_seconds() / 60.0
        if age_min < float(min_age_minutes):
            skipped += 1
            continue

        ticker = row.get("ticker")
        if not ticker:
            skipped += 1
            if logger:
                logger.debug(f"Paper resolver skip id={row.get('id')} reason=no_ticker")
            continue

        # Pull current mark
        try:
            raw_market = client.get_market(str(ticker))
            market_obj = raw_market
            if isinstance(raw_market, dict) and "market" in raw_market and isinstance(raw_market["market"], dict):
                market_obj = raw_market["market"]

            s = summarize_market(market_obj)
            quality = _extract_minimal_market_quality(s)

            mark_yes = quality["mid_yes_cents"]
            market_category = str(s.get("category", "")) if s.get("category") is not None else None

            if mark_yes is None:
                skipped += 1
                if logger:
                    logger.debug(f"Paper resolver skip {ticker} reason=no_mark")
                continue

            # Avoid resolving on trash marks (wide or dead markets)
            spread = quality["spread_cents"]
            volume = quality["volume"]
            status = quality["status"]

            if spread is not None and spread > max_resolve_spread_cents:
                skipped += 1
                if logger:
                    logger.debug(
                        f"Paper resolver skip {ticker} reason=wide_spread spread={spread}>{max_resolve_spread_cents}"
                    )
                continue

            if volume is not None and volume < min_resolve_volume:
                skipped += 1
                if logger:
                    logger.debug(
                        f"Paper resolver skip {ticker} reason=low_volume volume={volume}<{min_resolve_volume}"
                    )
                continue

            # If clearly closed/resolved, that's fine; if still trading, also fine (mark-to-market model).
            # We intentionally do NOT require final market resolution here.
            _ = status

        except Exception as e:
            if logger:
                logger.warning(f"Paper resolver market fetch failed for {ticker}: {e}")
            skipped += 1
            continue

        direction = str(row.get("direction", "SKIP")).upper()
        stake_dollars = float(row.get("stake_dollars", 0.0))

        # Try exact simulated entry price first; fallback to estimate.
        reasons = str(row.get("reasons", ""))
        entry_yes = _extract_entry_price_cents_from_reasons(reasons)
        entry_source = "logged"
        if entry_yes is None:
            entry_yes = _estimate_entry_price_from_decision(row)
            entry_source = "estimated"

        if entry_yes is None or direction not in {"YES", "NO"} or stake_dollars <= 0:
            skipped += 1
            if logger:
                logger.debug(
                    f"Paper resolver skip id={row.get('id')} reason=bad_trade_fields "
                    f"dir={direction} stake={stake_dollars} entry_yes={entry_yes}"
                )
            continue

        pnl = _paper_pnl_from_mark(
            direction=direction,
            stake_dollars=stake_dollars,
            entry_yes_price_cents=entry_yes,
            mark_yes_price_cents=mark_yes,
        )

        # Don't overfit tiny noise moves into wins/losses
        if abs(pnl) < flat_pnl_epsilon:
            won_value = None
        else:
            won_value = bool(pnl > 0)

        try:
            store.update_decision_outcome(
                decision_id=int(row["id"]),
                realized_pnl=float(pnl),
                won=won_value,
                resolved_ts=now.isoformat(),
                market_category=market_category,
            )
            updated += 1

            if logger:
                logger.info(
                    f"Paper resolved id={row['id']} {ticker} dir={direction} "
                    f"entry_yes={entry_yes}({entry_source}) mark_yes={mark_yes} "
                    f"stake={stake_dollars:.2f} pnl={pnl:.4f} won={won_value}"
                )
        except Exception as e:
            if logger:
                logger.warning(f"Paper resolver update failed for decision id={row.get('id')}: {e}")
            skipped += 1

    return PaperResolutionResult(updated=updated, checked=checked, skipped=skipped)
