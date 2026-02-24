from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SocialSignalResult:
    bonus_score: float
    reasons: List[str]


def evaluate_social_signal(market: dict, enabled: bool, max_bonus: int) -> SocialSignalResult:
    if not enabled:
        return SocialSignalResult(0.0, ["social_disabled"])

    # Placeholder "public activity" style heuristics from available market fields
    # (No blind account following, no unsupported copy trading endpoints)
    reasons: List[str] = []
    bonus = 0.0

    vol = market.get("volume", 0.0)
    spread = market.get("spread_cents", 99)

    if vol > 10000:
        bonus += 4
        reasons.append("high_volume_interest")
    elif vol > 1000:
        bonus += 2
        reasons.append("moderate_volume_interest")

    if spread <= 3:
        bonus += 3
        reasons.append("microstructure_clean")
    elif spread <= 6:
        bonus += 1
        reasons.append("microstructure_ok")

    # anti-chasing
    mid = market.get("mid_yes_cents", 50)
    if mid < 3 or mid > 97:
        bonus -= 2
        reasons.append("anti_chase_extreme")

    bonus = max(0.0, min(float(max_bonus), bonus))
    if not reasons:
        reasons.append("no_social_edge")
    return SocialSignalResult(bonus, reasons)
