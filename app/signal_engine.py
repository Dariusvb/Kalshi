from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SignalDecision:
    direction: str   # YES / NO / SKIP
    conviction_score: float
    edge_estimate_bps: float
    reasons: List[str]


def _f(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def evaluate_signal(market: dict, min_conviction: int, max_spread_cents: int, min_volume: int) -> SignalDecision:
    """
    Heuristic signal engine based on market microstructure / pricing only.

    Notes:
    - Does NOT read semantic market descriptions/questions.
    - Safe against missing keys / relaxed fallback summaries.
    - Keeps the same output interface used by main.py.
    """
    reasons: List[str] = []
    score = 50.0  # neutral starting point (slightly aggressive overall)

    # Safe reads (support stricter + relaxed picker summaries)
    spread = _f(market.get("spread_cents"), 999.0)
    volume = _f(market.get("volume"), 0.0)
    mid = _f(market.get("mid_yes_cents"), 50.0)
    yes_bid = _f(market.get("yes_bid"), 0.0)
    yes_ask = _f(market.get("yes_ask"), 0.0)

    # Optional flags added by relaxed chooser path in main.py
    pick_mode = str(market.get("_pick_mode", "") or "").lower()
    is_relaxed = bool(market.get("_relaxed")) or (pick_mode == "relaxed")

    # Basic quote sanity: if no usable spread and no two-sided quote, skip safely
    if spread <= 0 and not (yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid):
        reasons.append("bad_quote")
        return SignalDecision("SKIP", 0.0, 0.0, reasons + [f"below_threshold:{min_conviction}"])

    # Spread contribution
    if spread <= float(max_spread_cents):
        score += 12
        reasons.append(f"tight_spread:{int(round(spread))}")
    else:
        score -= min(20.0, spread)
        reasons.append(f"wide_spread:{int(round(spread))}")

    # Volume contribution
    if volume >= float(min_volume):
        score += 8
        reasons.append(f"volume_ok:{volume:.0f}")
    else:
        score -= 8
        reasons.append("low_volume")

    # New: slight penalty for relaxed fallback candidates so strict candidates remain preferred in score
    # (does not change strategy direction logic, only modestly reduces conviction)
    if is_relaxed:
        score -= 4
        reasons.append("relaxed_pick_penalty")

    # Simple edge heuristic:
    # prefer reversion away from extremes if spreads are sane
    direction = "SKIP"
    edge_bps = 0.0

    if 8 <= mid <= 35:
        direction = "YES"
        score += 10
        edge_bps = 60
        reasons.append("cheap_yes_zone")
    elif 65 <= mid <= 92:
        direction = "NO"
        score += 10
        edge_bps = 60
        reasons.append("expensive_yes_zone")
    elif 40 <= mid <= 60:
        score -= 5
        reasons.append("coinflip_zone")
    else:
        # very extreme
        score -= 4
        reasons.append("extreme_zone")

    score = max(0.0, min(100.0, score))

    if direction == "SKIP" or score < float(min_conviction):
        return SignalDecision("SKIP", score, edge_bps, reasons + [f"below_threshold:{min_conviction}"])

    return SignalDecision(direction, score, edge_bps, reasons)
