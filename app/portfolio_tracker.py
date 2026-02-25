from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PortfolioSnapshot:
    cash_balance_dollars: float = 0.0
    open_exposure_dollars: float = 0.0
    open_positions_count: int = 0
    daily_pnl_dollars: float = 0.0


def _to_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip().replace(",", "")
            if not s:
                return None
            return float(s)
    except Exception:
        return None
    return None


def _as_dollars_from_maybe_cents(value: float, *, force_cents: bool = False) -> float:
    """
    Heuristic conversion:
    - if force_cents=True, always divide by 100
    - otherwise, large values are assumed to be cents
    """
    if force_cents:
        return round(value / 100.0, 2)

    # Heuristic: values above 10,000 are more likely cents than dollars
    # (10,001 dollars cash is possible, but this is a best-effort fallback)
    if abs(value) > 10000:
        return round(value / 100.0, 2)

    return round(value, 2)


def parse_balance_payload(payload: dict) -> float:
    """
    Best-effort normalization for portfolio/cash balance payloads.

    Supports:
    - flat numeric fields
    - numeric strings
    - common cents field names
    - one-level nested dicts (e.g., {"balance": {"available_funds": 12345}})
    """
    if not isinstance(payload, dict):
        return 0.0

    # Prefer "available to trade" style fields first if present
    dollar_keys = (
        "available_funds",
        "available_cash",
        "cash_balance",
        "cash",
        "balance",
    )
    cents_keys = (
        "available_funds_cents",
        "available_cash_cents",
        "cash_balance_cents",
        "cash_cents",
        "balance_cents",
    )

    # 1) Direct cents keys (most explicit)
    for key in cents_keys:
        if key in payload:
            v = _to_float(payload.get(key))
            if v is not None:
                return _as_dollars_from_maybe_cents(v, force_cents=True)

    # 2) Direct dollar-ish keys
    for key in dollar_keys:
        if key in payload:
            raw = payload.get(key)

            # If nested dict, inspect one level down
            if isinstance(raw, dict):
                for subkey in cents_keys + dollar_keys:
                    if subkey in raw:
                        sv = _to_float(raw.get(subkey))
                        if sv is not None:
                            return _as_dollars_from_maybe_cents(
                                sv, force_cents=subkey.endswith("_cents")
                            )
            else:
                v = _to_float(raw)
                if v is not None:
                    return _as_dollars_from_maybe_cents(v)

    # 3) Generic fallback: look for any "*_cents" numeric field first
    for k, vraw in payload.items():
        if isinstance(k, str) and k.endswith("_cents"):
            v = _to_float(vraw)
            if v is not None:
                return _as_dollars_from_maybe_cents(v, force_cents=True)

    # 4) Last resort: first numeric-looking value among common-ish cash words
    for k, vraw in payload.items():
        if not isinstance(k, str):
            continue
        lk = k.lower()
        if any(word in lk for word in ("cash", "balance", "funds", "available")):
            v = _to_float(vraw)
            if v is not None:
                return _as_dollars_from_maybe_cents(v)

    return 0.0
