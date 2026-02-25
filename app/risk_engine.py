from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    stake_dollars: float
    reason: str


def position_size_from_conviction(
    conviction: float,
    min_dollars: int,
    max_dollars: int,
    social_size_boost_pct: float = 0.0,
    learning_multiplier: float = 1.0,
) -> float:
    # More risk-on sizing curve but capped
    base = min_dollars + (max_dollars - min_dollars) * (conviction / 100.0)

    boosted = base * (1.0 + max(0.0, social_size_boost_pct))
    boosted *= max(0.5, learning_multiplier)  # additional safety floor

    return round(min(max_dollars, max(min_dollars, boosted)), 2)

def evaluate_risk(
    *,
    conviction: float,
    current_open_exposure: float,
    open_positions_count: int,
    daily_pnl: float,
    min_dollars: int,
    max_dollars: int,
    daily_max_loss_dollars: float,
    max_open_exposure_dollars: float,
    max_simultaneous_positions: int,
    social_size_boost_pct: float = 0.0,
    learning_multiplier: float = 1.0,
) -> RiskDecision:
    
    if daily_pnl <= -abs(daily_max_loss_dollars):
        return RiskDecision(False, 0.0, "daily_loss_cap_hit")

    if open_positions_count >= max_simultaneous_positions:
        return RiskDecision(False, 0.0, "max_positions_hit")

    stake = position_size_from_conviction(conviction, min_dollars, max_dollars, social_size_boost_pct, learning_multiplier)

    if current_open_exposure + stake > max_open_exposure_dollars:
        return RiskDecision(False, 0.0, "max_open_exposure_hit")

    return RiskDecision(True, stake, "ok")
