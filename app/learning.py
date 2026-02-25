from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LearningAdjustment:
    stake_multiplier: float
    conviction_adjustment: float
    mode: str
    reasons: list[str]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _is_trade(row: dict) -> bool:
    return str(row.get("action", "")).startswith("TRADE")


def _is_resolved_trade(row: dict) -> bool:
    return _is_trade(row) and row.get("resolved_ts") is not None


def _won_value(row: dict) -> Optional[int]:
    """
    Normalize won field to 1/0/None.
    """
    v = row.get("won")
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        if v in (0, 1):
            return v
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "won", "win"}:
            return 1
        if s in {"0", "false", "lost", "loss"}:
            return 0
    return None


def _rolling_drawdown_proxy(resolved_trades_desc: List[dict]) -> float:
    """
    Approximate drawdown using realized_pnl in chronological order.
    Input expected newest-first; we reverse to oldest-first.
    Returns max drawdown in dollars over the provided window.
    """
    rows = list(reversed(resolved_trades_desc))
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for r in rows:
        pnl = r.get("realized_pnl")
        if pnl is None:
            continue
        equity += _safe_float(pnl)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return round(max_dd, 4)


def _loss_streak(resolved_trades_desc: List[dict]) -> int:
    """
    Count consecutive losses from most recent backward.
    """
    streak = 0
    for r in resolved_trades_desc:
        w = _won_value(r)
        if w is None:
            continue
        if w == 0:
            streak += 1
        else:
            break
    return streak


def _category_bias_info(resolved_trades_desc: List[dict]) -> tuple[Optional[str], float, str]:
    """
    Lightweight category performance check.
    Returns:
      (best_category_or_none, category_multiplier_hint, reason)
    Small influence only.
    """
    stats: dict[str, dict[str, float]] = {}

    for r in resolved_trades_desc:
        cat = r.get("market_category")
        if not cat:
            continue
        cat = str(cat).strip().lower()
        if not cat:
            continue

        stats.setdefault(cat, {"n": 0.0, "pnl": 0.0, "wins": 0.0, "losses": 0.0})
        stats[cat]["n"] += 1

        pnl = r.get("realized_pnl")
        if pnl is not None:
            stats[cat]["pnl"] += _safe_float(pnl)

        w = _won_value(r)
        if w == 1:
            stats[cat]["wins"] += 1
        elif w == 0:
            stats[cat]["losses"] += 1

    if not stats:
        return None, 1.0, "no_category_data"

    eligible = {k: v for k, v in stats.items() if v["n"] >= 3}
    if not eligible:
        return None, 1.0, "category_sample_too_small"

    best_cat = max(
        eligible.items(),
        key=lambda kv: (kv[1]["pnl"], kv[1]["wins"] - kv[1]["losses"])
    )
    cat, m = best_cat

    # Tiny influence to avoid overfitting categories
    if m["pnl"] > 0:
        return cat, 1.03, f"category_positive:{cat}"
    if m["pnl"] < 0:
        return cat, 0.98, f"category_negative:{cat}"
    return cat, 1.0, f"category_neutral:{cat}"


def _weighted_outcome_metrics(outcome_rows_desc: List[dict]) -> dict:
    """
    Newest-first rows.
    Recency-weighted metrics so recent results matter more without fully ignoring history.
    """
    # weights decay gently: 1.0, 0.93, 0.86...
    weight = 1.0
    decay = 0.93

    w_wins = 0.0
    w_losses = 0.0
    w_pnl_sum = 0.0
    w_pnl_weight = 0.0

    classified_count = 0
    pnl_count = 0

    for r in outcome_rows_desc:
        wv = _won_value(r)
        if wv == 1:
            w_wins += weight
            classified_count += 1
        elif wv == 0:
            w_losses += weight
            classified_count += 1

        pnl = r.get("realized_pnl")
        if pnl is not None:
            p = _safe_float(pnl)
            w_pnl_sum += p * weight
            w_pnl_weight += weight
            pnl_count += 1

        weight *= decay

    weighted_wr = None
    if (w_wins + w_losses) > 0:
        weighted_wr = w_wins / (w_wins + w_losses)

    weighted_expectancy = None
    if w_pnl_weight > 0:
        weighted_expectancy = w_pnl_sum / w_pnl_weight

    return {
        "weighted_win_rate": weighted_wr,
        "weighted_expectancy": weighted_expectancy,
        "classified_count": classified_count,
        "pnl_count": pnl_count,
    }


def compute_learning_adjustment(recent_decisions: List[dict]) -> LearningAdjustment:
    """
    Adaptive sizing/threshold logic.

    Priority:
    1) Use resolved trade outcomes (won/realized_pnl/resolved_ts) if available
    2) Fall back to proxy behavior (conviction/skips/trade frequency) if not enough outcomes

    Output:
      - stake_multiplier: scales stake sizing (risk engine still enforces caps)
      - conviction_adjustment: negative loosens threshold, positive tightens threshold
      - mode: human-readable mode string
      - reasons: audit trail for Discord/logging
    """
    if not recent_decisions:
        return LearningAdjustment(1.0, 0.0, "neutral", ["no_history"])

    reasons: list[str] = []

    trades = [d for d in recent_decisions if _is_trade(d)]
    skips = [d for d in recent_decisions if d.get("action") == "SKIP"]
    resolved = [d for d in recent_decisions if _is_resolved_trade(d)]

    if len(trades) < 5:
        return LearningAdjustment(1.0, 0.0, "warmup", [f"few_trades:{len(trades)}"])

    avg_conv = sum(_safe_float(d.get("final_conviction", 0.0)) for d in trades) / max(1, len(trades))
    avg_stake = sum(_safe_float(d.get("stake_dollars", 0.0)) for d in trades) / max(1, len(trades))

    # Defaults
    stake_multiplier = 1.0
    conviction_adjustment = 0.0
    mode = "neutral"

    # -----------------------------------------------------------
    # A) Outcome-aware learning (preferred)
    # -----------------------------------------------------------
    outcome_rows = [
        r for r in resolved
        if (_won_value(r) is not None) or (r.get("realized_pnl") is not None)
    ]

    if len(outcome_rows) >= 5:
        wins = 0
        losses = 0
        pnl_vals: list[float] = []

        for r in outcome_rows:
            w = _won_value(r)
            if w == 1:
                wins += 1
            elif w == 0:
                losses += 1

            if r.get("realized_pnl") is not None:
                pnl_vals.append(_safe_float(r.get("realized_pnl")))

        total_classified = wins + losses
        win_rate = (wins / total_classified) if total_classified > 0 else None
        rolling_pnl = sum(pnl_vals) if pnl_vals else 0.0
        expectancy = (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else 0.0
        dd_proxy = _rolling_drawdown_proxy(outcome_rows)
        loss_streak = _loss_streak(outcome_rows)

        weighted = _weighted_outcome_metrics(outcome_rows)
        w_wr = weighted["weighted_win_rate"]
        w_exp = weighted["weighted_expectancy"]

        reasons.append(f"resolved={len(outcome_rows)}")
        if win_rate is not None:
            reasons.append(f"wr={win_rate*100:.1f}%")
        if w_wr is not None:
            reasons.append(f"w_wr={w_wr*100:.1f}%")
        reasons.append(f"pnl={rolling_pnl:.2f}")
        reasons.append(f"exp={expectancy:.3f}")
        if w_exp is not None:
            reasons.append(f"w_exp={w_exp:.3f}")
        reasons.append(f"dd={dd_proxy:.2f}")
        reasons.append(f"loss_streak={loss_streak}")

        # Category hint (small influence)
        _, cat_mult_hint, cat_reason = _category_bias_info(outcome_rows)
        reasons.append(cat_reason)

        # Blend raw + weighted for smoother behavior
        # If one metric is missing, fall back to the one that exists.
        eff_wr = None
        if win_rate is not None and w_wr is not None:
            eff_wr = (0.45 * win_rate) + (0.55 * w_wr)
        elif w_wr is not None:
            eff_wr = w_wr
        elif win_rate is not None:
            eff_wr = win_rate

        eff_exp = None
        if pnl_vals and w_exp is not None:
            eff_exp = (0.40 * expectancy) + (0.60 * w_exp)
        elif pnl_vals:
            eff_exp = expectancy
        elif w_exp is not None:
            eff_exp = w_exp

        if eff_wr is not None:
            reasons.append(f"eff_wr={eff_wr*100:.1f}%")
        if eff_exp is not None:
            reasons.append(f"eff_exp={eff_exp:.3f}")

        # ---- De-risk conditions first (moderate, not excessive) ----
        # Multipliers compound, so keep each change small/moderate.
        if loss_streak >= 4:
            stake_multiplier *= 0.78
            conviction_adjustment += 4.0
            mode = "de_risk_loss_streak"
            reasons.append("4+ consecutive losses")
        elif loss_streak == 3:
            stake_multiplier *= 0.86
            conviction_adjustment += 2.5
            mode = "de_risk_loss_streak"
            reasons.append("3 consecutive losses")

        if eff_wr is not None and total_classified >= 6 and eff_wr < 0.44:
            stake_multiplier *= 0.88
            conviction_adjustment += 2.5
            if mode == "neutral":
                mode = "de_risk_low_winrate"
            reasons.append("low_effective_win_rate")

        if eff_exp is not None and eff_exp < 0:
            stake_multiplier *= 0.92
            conviction_adjustment += 1.5
            if mode == "neutral":
                mode = "de_risk_negative_expectancy"
            reasons.append("negative_effective_expectancy")

        # Drawdown threshold scales with observed stake so it doesn't choke small accounts
        dd_threshold = max(12.0, avg_stake * 2.0)
        if dd_proxy >= dd_threshold:
            stake_multiplier *= 0.92
            conviction_adjustment += 1.5
            if mode == "neutral":
                mode = "de_risk_drawdown"
            reasons.append("drawdown_proxy_elevated")

        # ---- Controlled risk-on (only when truly earning it) ----
        # This is intentionally easier than "perfect regime", so it still stands on business.
        can_risk_on = True
        if loss_streak >= 3:
            can_risk_on = False
        if eff_exp is not None and eff_exp <= 0:
            can_risk_on = False
        if eff_wr is not None and total_classified >= 6 and eff_wr < 0.50:
            can_risk_on = False

        if can_risk_on and total_classified >= 8:
            # Base risk-on if decent regime
            if eff_wr is not None and eff_wr >= 0.56 and (eff_exp is None or eff_exp > 0):
                stake_multiplier *= 1.08
                conviction_adjustment -= 1.0
                mode = "risk_on_outcomes"
                reasons.append("good_effective_wr_expectancy")

            # Stronger but still bounded
            if (
                eff_wr is not None
                and eff_wr >= 0.62
                and (eff_exp is not None and eff_exp > 0)
                and dd_proxy < max(8.0, avg_stake * 1.2)
            ):
                stake_multiplier *= 1.05
                conviction_adjustment -= 0.5
                mode = "risk_on_strong"
                reasons.append("strong_outcome_regime")

        # Apply tiny category influence at the end
        stake_multiplier *= cat_mult_hint

    # -----------------------------------------------------------
    # B) Proxy fallback / supplement (when outcome data is thin)
    # -----------------------------------------------------------
    else:
        reasons.append(f"resolved={len(outcome_rows)}")
        reasons.append("using_proxy_logic")

        # Risk-on proxy is mild; avoid overconfidence
        if len(trades) >= 10 and avg_conv >= 68:
            stake_multiplier += 0.08
            mode = "risk_on_proxy"
            reasons.append("high_avg_conviction")

        if len(skips) > len(trades) * 2:
            # Too many skips means we may be filtering too hard; loosen a bit.
            conviction_adjustment -= 2.0
            reasons.append("too_many_skips_loosen_threshold")

        # If conviction is low despite many trades, tighten slightly
        if len(trades) >= 10 and avg_conv < 55:
            conviction_adjustment += 1.5
            if mode == "neutral":
                mode = "tighten_proxy"
            reasons.append("low_avg_conviction")

        # If it's trading a lot with almost no skips, nudge threshold up a bit to avoid spray-and-pray
        if len(trades) >= 12 and len(skips) <= 1:
            conviction_adjustment += 1.0
            if mode == "neutral":
                mode = "discipline_proxy"
            reasons.append("very_low_skips_many_trades")

    # -----------------------------------------------------------
    # C) Global caps / sanity
    # -----------------------------------------------------------
    # Bounded but not too tight (lets it express adaptation).
    stake_multiplier = max(0.72, min(1.32, stake_multiplier))
    conviction_adjustment = max(-5.0, min(6.0, conviction_adjustment))

    if not reasons:
        reasons.append("stable")

    reasons.append(f"trades={len(trades)}")
    reasons.append(f"skips={len(skips)}")
    reasons.append(f"avg_conv={avg_conv:.1f}")
    reasons.append(f"avg_stake={avg_stake:.2f}")

    return LearningAdjustment(
        stake_multiplier=stake_multiplier,
        conviction_adjustment=conviction_adjustment,
        mode=mode,
        reasons=reasons,
    )
