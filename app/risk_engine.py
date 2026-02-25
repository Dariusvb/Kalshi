from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    stake_dollars: float
    reason: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def position_size_from_conviction(
    conviction: float,
    min_dollars: int,
    max_dollars: int,
    social_size_boost_pct: float = 0.0,
    learning_multiplier: float = 1.0,
    *,
    allow_risk_on_sizing: bool = True,
    recent_win_rate: float | None = None,
    min_win_rate_for_risk_on: float = 0.50,
) -> float:
    """
    Position sizing:
    - Base size scales with conviction.
    - Social boost is modest and capped upstream.
    - Learning multiplier can increase/decrease size, but only if risk-on is allowed.
    - If recent win rate is known and below threshold, risk-on sizing is disabled.

    NOTE:
    This function does not know about realized PnL yet; it only respects the inputs passed in.
    Once your learning module calculates rolling win rate / expectancy / drawdown, feed them here.
    """
    conviction = _clamp(float(conviction), 0.0, 100.0)

    # Base: linear sizing between min/max
    base = float(min_dollars) + (float(max_dollars) - float(min_dollars)) * (conviction / 100.0)

    # Social boost (already capped by caller/config)
    social_boost = 1.0 + max(0.0, float(social_size_boost_pct))
    sized = base * social_boost

    # Learning multiplier rules
    lm = float(learning_multiplier)

    # If we know win rate and it's below threshold, disable risk-on increases
    if recent_win_rate is not None and recent_win_rate < float(min_win_rate_for_risk_on):
        lm = min(lm, 1.0)

    # If risk-on sizing disabled globally/by current state, no learning-based increase allowed
    if not allow_risk_on_sizing:
        lm = min(lm, 1.0)

    # Safety clamps:
    # - allow reductions down to 0.5x (de-risk)
    # - allow increases only up to 1.35x here (even if learning says more)
    lm = _clamp(lm, 0.5, 1.35)

    sized *= lm

    return round(_clamp(sized, float(min_dollars), float(max_dollars)), 2)


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
    # ---- New optional controls for future "real learning" integration ----
    drawdown_lockout: bool = False,
    loss_streak_cooldown: bool = False,
    category_exposure_dollars: float | None = None,
    max_category_exposure_dollars: float | None = None,
    recent_win_rate: float | None = None,           # 0.0 to 1.0, if available
    min_win_rate_for_risk_on: float = 0.50,         # risk-on only if rolling win rate >= this
    available_cash_dollars: float | None = None,    # optional cash gate for live safety
) -> RiskDecision:
    """
    Central risk gate.

    Current hard stops:
    - daily loss cap
    - max simultaneous positions
    - max open exposure
    - optional available cash check

    Future-ready soft/optional stops:
    - drawdown lockout
    - loss streak cooldown
    - category exposure cap
    - win-rate-aware risk-on gating

    Returns:
        RiskDecision(allowed, stake_dollars, reason)
    """
    # ---- Hard lockouts first ----
    if drawdown_lockout:
        return RiskDecision(False, 0.0, "drawdown_lockout")

    if loss_streak_cooldown:
        return RiskDecision(False, 0.0, "loss_streak_cooldown")

    if float(daily_pnl) <= -abs(float(daily_max_loss_dollars)):
        return RiskDecision(False, 0.0, "daily_loss_cap_hit")

    if int(open_positions_count) >= int(max_simultaneous_positions):
        return RiskDecision(False, 0.0, "max_positions_hit")

    # Optional early cash guard (useful in live mode to avoid futile order attempts)
    if available_cash_dollars is not None and float(available_cash_dollars) < float(min_dollars):
        return RiskDecision(False, 0.0, "insufficient_cash_min_trade")

    # Determine whether learning/social risk-on is allowed
    allow_risk_on_sizing = True
    if recent_win_rate is not None and recent_win_rate < float(min_win_rate_for_risk_on):
        allow_risk_on_sizing = False

    # Stake sizing (includes conviction/social/learning effects)
    stake = position_size_from_conviction(
        conviction=conviction,
        min_dollars=min_dollars,
        max_dollars=max_dollars,
        social_size_boost_pct=social_size_boost_pct,
        learning_multiplier=learning_multiplier,
        allow_risk_on_sizing=allow_risk_on_sizing,
        recent_win_rate=recent_win_rate,
        min_win_rate_for_risk_on=min_win_rate_for_risk_on,
    )

    # Optional cash gate (prevents placing an order bigger than available cash)
    if available_cash_dollars is not None and float(available_cash_dollars) < float(stake):
        return RiskDecision(False, 0.0, "insufficient_cash")

    # ---- Exposure caps ----
    if float(current_open_exposure) + float(stake) > float(max_open_exposure_dollars):
        return RiskDecision(False, 0.0, "max_open_exposure_hit")

    if (
        category_exposure_dollars is not None
        and max_category_exposure_dollars is not None
        and float(category_exposure_dollars) + float(stake) > float(max_category_exposure_dollars)
    ):
        return RiskDecision(False, 0.0, "max_category_exposure_hit")

    # ---- Informative reason tags ----
    if not allow_risk_on_sizing and learning_multiplier > 1.0:
        return RiskDecision(True, stake, "ok_risk_on_suppressed")

    return RiskDecision(True, stake, "ok")
