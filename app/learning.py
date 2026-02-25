from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LearningAdjustment:
    stake_multiplier: float
    conviction_adjustment: float
    mode: str
    reasons: list[str]


def compute_learning_adjustment(recent_decisions: List[dict]) -> LearningAdjustment:
    """
    Lightweight adaptive logic using recent logged decisions.
    NOTE: In current MVP, decisions don't yet include realized PnL, so this uses proxy signals.
    As you add fills/results, upgrade this to use realized outcomes.
    """
    if not recent_decisions:
        return LearningAdjustment(1.0, 0.0, "neutral", ["no_history"])

    reasons: list[str] = []
    trades = [d for d in recent_decisions if str(d.get("action", "")).startswith("TRADE")]
    skips = [d for d in recent_decisions if d.get("action") == "SKIP"]

    if len(trades) < 5:
        return LearningAdjustment(1.0, 0.0, "warmup", [f"few_trades:{len(trades)}"])

    avg_conv = sum(float(d.get("final_conviction", 0)) for d in trades) / len(trades)
    avg_stake = sum(float(d.get("stake_dollars", 0)) for d in trades) / len(trades)

    # Proxy-based adaptation (until realized PnL is wired in)
    # If system is trading often with strong conviction, allow modest risk-on.
    stake_multiplier = 1.0
    conviction_adjustment = 0.0
    mode = "neutral"

    if len(trades) >= 10 and avg_conv >= 68:
        stake_multiplier += 0.10
        mode = "risk_on"
        reasons.append("high_avg_conviction")

    if len(skips) > len(trades) * 2:
        # Too many skips means we're filtering a lot; slightly loosen threshold.
        conviction_adjustment -= 2.0
        reasons.append("too_many_skips_loosen_threshold")

    # Hard caps for safety
    stake_multiplier = max(0.75, min(1.25, stake_multiplier))

    if not reasons:
        reasons.append("stable")
    reasons.append(f"trades={len(trades)}")
    reasons.append(f"avg_conv={avg_conv:.1f}")
    reasons.append(f"avg_stake={avg_stake:.2f}")

    return LearningAdjustment(
        stake_multiplier=stake_multiplier,
        conviction_adjustment=conviction_adjustment,
        mode=mode,
        reasons=reasons,
    )
