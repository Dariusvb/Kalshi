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

    Design goals:
    - Allow more real dry-run trades in the market regimes you're actually seeing.
    - Still avoid stupid trades on broken quotes / coinflip zones / ultra-bad tails.
    - Be moderately aggressive, but with microstructure gating in extreme tails.
    """
    reasons: List[str] = []
    score = 50.0  # neutral baseline (slightly aggressive overall)

    # Safe reads (support strict + relaxed picker summaries)
    spread = _f(market.get("spread_cents"), 999.0)
    volume = _f(market.get("volume"), 0.0)
    mid = _f(market.get("mid_yes_cents"), 50.0)
    yes_bid = _f(market.get("yes_bid"), 0.0)
    yes_ask = _f(market.get("yes_ask"), 0.0)

    # Optional flags added by relaxed chooser path in main.py
    pick_mode = str(market.get("_pick_mode", "") or "").lower()
    is_relaxed = bool(market.get("_relaxed")) or (pick_mode == "relaxed")

    # -------------------------------------------------------------------------
    # Quote sanity / fallback spread inference (safe with one-sided or relaxed quotes)
    # -------------------------------------------------------------------------
    inferred_spread = None
    if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
        inferred_spread = float(yes_ask - yes_bid)

    # If spread is missing/bad but we have a valid two-sided quote, use inferred spread.
    # If we have only one-sided quotes, keep the reported spread if present; otherwise skip.
    effective_spread = spread
    if effective_spread <= 0:
        if inferred_spread is not None:
            effective_spread = inferred_spread
            reasons.append(f"inferred_spread:{int(round(effective_spread))}")
        else:
            reasons.append("bad_quote")
            return SignalDecision("SKIP", 0.0, 0.0, reasons + [f"below_threshold:{int(min_conviction)}"])

    # Mid sanity (protect against broken summaries)
    if mid <= 0 or mid >= 100:
        reasons.append("invalid_mid")
        return SignalDecision("SKIP", 0.0, 0.0, reasons + [f"below_threshold:{int(min_conviction)}"])

    # -------------------------------------------------------------------------
    # Microstructure scoring
    # -------------------------------------------------------------------------
    # Spread contribution (slightly more nuanced than before)
    if effective_spread <= float(max_spread_cents):
        score += 12
        reasons.append(f"tight_spread:{int(round(effective_spread))}")
    elif effective_spread <= float(max_spread_cents) + 2:
        score += 4
        reasons.append(f"ok_spread:{int(round(effective_spread))}")
    else:
        # Penalize but don't insta-kill unless very wide
        score -= min(22.0, effective_spread)
        reasons.append(f"wide_spread:{int(round(effective_spread))}")

    # Volume contribution (a bit less punitive than before so good tails can pass)
    if volume >= float(min_volume):
        score += 8
        reasons.append(f"volume_ok:{volume:.0f}")
    else:
        score -= 6
        reasons.append("low_volume")

    # Liquidity bonuses (helps quality tails pass conviction)
    # Keep these small so they don't overpower bad spreads.
    strong_vol_1 = max(float(min_volume) * 3.0, 150.0)
    strong_vol_2 = max(float(min_volume) * 8.0, 600.0)
    if volume >= strong_vol_1:
        score += 3
        reasons.append("volume_strong")
    if volume >= strong_vol_2:
        score += 2
        reasons.append("volume_very_strong")

    # Relaxed chooser candidates should still be tradable, but slightly lower confidence.
    if is_relaxed:
        score -= 3
        reasons.append("relaxed_pick_penalty")

    # -------------------------------------------------------------------------
    # Price-zone / edge heuristic
    # -------------------------------------------------------------------------
    # We keep the same broad reversion logic, but give extreme tails a conditional path.
    direction = "SKIP"
    edge_bps = 0.0

    # Core zones (high confidence)
    if 8 <= mid <= 35:
        direction = "YES"
        score += 12
        edge_bps = 60.0
        reasons.append("cheap_yes_zone")

    elif 65 <= mid <= 92:
        direction = "NO"
        score += 12
        edge_bps = 60.0
        reasons.append("expensive_yes_zone")

    # Near-edge zones (slightly weaker but still actionable)
    elif 5 <= mid < 8:
        direction = "YES"
        score += 8
        edge_bps = 45.0
        reasons.append("near_cheap_yes_zone")

    elif 92 < mid <= 95:
        direction = "NO"
        score += 8
        edge_bps = 45.0
        reasons.append("near_expensive_yes_zone")

    # Transition zones (not ideal, but not auto-trash)
    elif 35 < mid < 40:
        score -= 3
        reasons.append("upper_cheap_transition")

    elif 60 < mid < 65:
        score -= 3
        reasons.append("lower_expensive_transition")

    # Coinflip zone: still mostly avoid
    elif 40 <= mid <= 60:
        score -= 8
        reasons.append("coinflip_zone")

    # Extreme tails: allow conditional participation, but require decent microstructure
    elif 2 <= mid < 5:
        direction = "YES"
        edge_bps = 25.0
        reasons.append("extreme_zone")
        reasons.append("extreme_yes_tail")

        # Tail base score is modest; quality determines if it survives threshold
        score += 4

        # Tail-specific microstructure gate
        tail_spread_ok = effective_spread <= float(max_spread_cents)
        tail_volume_ok = volume >= float(min_volume)
        if tail_spread_ok and tail_volume_ok:
            score += 8
            reasons.append("extreme_microstructure_ok")
        elif tail_spread_ok or tail_volume_ok:
            score += 1
            reasons.append("extreme_microstructure_mixed")
        else:
            score -= 10
            reasons.append("extreme_microstructure_weak")

    elif 95 < mid <= 98:
        direction = "NO"
        edge_bps = 25.0
        reasons.append("extreme_zone")
        reasons.append("extreme_no_tail")

        score += 4

        tail_spread_ok = effective_spread <= float(max_spread_cents)
        tail_volume_ok = volume >= float(min_volume)
        if tail_spread_ok and tail_volume_ok:
            score += 8
            reasons.append("extreme_microstructure_ok")
        elif tail_spread_ok or tail_volume_ok:
            score += 1
            reasons.append("extreme_microstructure_mixed")
        else:
            score -= 10
            reasons.append("extreme_microstructure_weak")

    # Ultra tails / likely broken pricing
    else:
        score -= 18
        reasons.append("untradeable_tail")

    # -------------------------------------------------------------------------
    # Additional guardrails (don't be stupid)
    # -------------------------------------------------------------------------
    # Refuse obviously too-wide spreads regardless of zone.
    # (Small buffer above configured max keeps some flexibility but blocks junk.)
    hard_spread_cap = max(float(max_spread_cents) + 4.0, float(max_spread_cents) * 1.6)
    if effective_spread > hard_spread_cap:
        reasons.append("spread_hard_cap")
        direction = "SKIP"
        edge_bps = 0.0
        score = min(score, float(min_conviction) - 1.0)

    # If we somehow got a direction but quote quality is too weak in relaxed mode, nudge down.
    # This prevents relaxed one-sided junk from passing too easily.
    if is_relaxed and inferred_spread is None and yes_bid <= 0:
        score -= 4
        reasons.append("relaxed_one_sided_penalty")

    # Clamp final score
    score = max(0.0, min(100.0, score))

    # Final gate
    if direction == "SKIP" or score < float(min_conviction):
        return SignalDecision(
            "SKIP",
            score,
            edge_bps,
            reasons + [f"below_threshold:{int(min_conviction)}"]
        )

    return SignalDecision(direction, score, edge_bps, reasons)
