from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SignalDecision:
    direction: str   # YES / NO / SKIP
    conviction_score: float
    edge_estimate_bps: float
    reasons: List[str]


def evaluate_signal(market: dict, min_conviction: int, max_spread_cents: int, min_volume: int) -> SignalDecision:
    reasons: List[str] = []
    score = 50.0  # neutral starting point (slightly aggressive overall)
    spread = market["spread_cents"]
    volume = market["volume"]
    mid = market["mid_yes_cents"]

    if spread <= max_spread_cents:
        score += 12
        reasons.append(f"tight_spread:{spread}")
    else:
        score -= min(20, spread)
        reasons.append(f"wide_spread:{spread}")

    if volume >= min_volume:
        score += 8
        reasons.append(f"volume_ok:{volume:.0f}")
    else:
        score -= 8
        reasons.append("low_volume")

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
    if direction == "SKIP" or score < min_conviction:
        return SignalDecision("SKIP", score, edge_bps, reasons + [f"below_threshold:{min_conviction}"])
    return SignalDecision(direction, score, edge_bps, reasons)
