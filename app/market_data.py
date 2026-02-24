from __future__ import annotations

from typing import Any, Dict, List


def extract_markets(raw: Any) -> List[dict]:
    """
    Normalize Kalshi markets payload into a list.
    Handles common wrapper patterns.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("markets", "data", "results"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def summarize_market(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort normalized fields; Kalshi payload fields can evolve.
    """
    ticker = m.get("ticker") or m.get("market_ticker") or m.get("id") or "UNKNOWN"
    yes_bid = m.get("yes_bid") or m.get("best_bid_yes") or m.get("bid_yes") or 0
    yes_ask = m.get("yes_ask") or m.get("best_ask_yes") or m.get("ask_yes") or 0

    # Fallback if only one side quote is available
    if not yes_ask and m.get("yes_price"):
        yes_ask = m.get("yes_price")
    if not yes_bid and m.get("yes_price"):
        yes_bid = m.get("yes_price")

    volume = m.get("volume") or m.get("volume_24h") or m.get("recent_volume") or 0
    status = m.get("status") or m.get("market_status") or "unknown"

    spread = abs(int(yes_ask or 0) - int(yes_bid or 0))
    mid = ((int(yes_ask or 0) + int(yes_bid or 0)) / 2) if (yes_ask or yes_bid) else 0

    return {
        "ticker": str(ticker),
        "yes_bid": int(yes_bid or 0),
        "yes_ask": int(yes_ask or 0),
        "spread_cents": int(spread),
        "mid_yes_cents": float(mid),
        "volume": float(volume or 0),
        "status": str(status),
        "raw": m,
    }
