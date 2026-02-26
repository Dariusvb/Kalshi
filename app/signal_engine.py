from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


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
    Microstructure-only signal.

    Patch goals (fix "only YES @ 6-8c" bias):
    - STOP buying ultra-cheap YES tails (the 2–10c longshot/parlay traps)
    - Evaluate BOTH sides (YES and NO) and choose the better one
    - Keep quote sanity checks and spread/volume gating
    """
    reasons: List[str] = []

    # Safe reads (support strict + relaxed picker summaries)
    spread = _f(market.get("spread_cents"), 999.0)
    volume = _f(market.get("volume"), 0.0)
    mid_yes = _f(market.get("mid_yes_cents"), 50.0)
    yes_bid = _f(market.get("yes_bid"), 0.0)
    yes_ask = _f(market.get("yes_ask"), 0.0)

    pick_mode = str(market.get("_pick_mode", "") or "").lower()
    is_relaxed = bool(market.get("_relaxed")) or (pick_mode == "relaxed")

    # -------------------------------------------------------------------------
    # Quote sanity / fallback spread inference
    # -------------------------------------------------------------------------
    inferred_spread = None
    if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
        inferred_spread = float(yes_ask - yes_bid)

    effective_spread = spread
    if effective_spread <= 0:
        if inferred_spread is not None:
            effective_spread = inferred_spread
            reasons.append(f"inferred_spread:{int(round(effective_spread))}")
        else:
            reasons.append("bad_quote")
            return SignalDecision("SKIP", 0.0, 0.0, reasons + [f"below_threshold:{int(min_conviction)}"])

    if mid_yes <= 0 or mid_yes >= 100:
        reasons.append("invalid_mid")
        return SignalDecision("SKIP", 0.0, 0.0, reasons + [f"below_threshold:{int(min_conviction)}"])

    # -------------------------------------------------------------------------
    # Base microstructure score (direction-agnostic)
    # -------------------------------------------------------------------------
    base = 50.0

    # Spread
    if effective_spread <= float(max_spread_cents):
        base += 12
        reasons.append(f"tight_spread:{int(round(effective_spread))}")
    elif effective_spread <= float(max_spread_cents) + 2:
        base += 4
        reasons.append(f"ok_spread:{int(round(effective_spread))}")
    else:
        base -= min(22.0, effective_spread)
        reasons.append(f"wide_spread:{int(round(effective_spread))}")

    # Volume
    if volume >= float(min_volume):
        base += 8
        reasons.append(f"volume_ok:{volume:.0f}")
    else:
        base -= 6
        reasons.append("low_volume")

    # Liquidity bonuses (small)
    strong_vol_1 = max(float(min_volume) * 3.0, 150.0)
    strong_vol_2 = max(float(min_volume) * 8.0, 600.0)
    if volume >= strong_vol_1:
        base += 3
        reasons.append("volume_strong")
    if volume >= strong_vol_2:
        base += 2
        reasons.append("volume_very_strong")

    if is_relaxed:
        base -= 3
        reasons.append("relaxed_pick_penalty")

    # One-sided relaxed quotes penalty (don’t let junk pass)
    if is_relaxed and inferred_spread is None and yes_bid <= 0:
        base -= 4
        reasons.append("relaxed_one_sided_penalty")

    # Hard spread cap (absolute “nope”)
    hard_spread_cap = max(float(max_spread_cents) + 4.0, float(max_spread_cents) * 1.6)
    if effective_spread > hard_spread_cap:
        reasons.append("spread_hard_cap")
        base = min(base, float(min_conviction) - 1.0)

    # -------------------------------------------------------------------------
    # Side selection (bidirectional)
    # -------------------------------------------------------------------------
    mid_no = 100.0 - mid_yes

    # These are the key bias-fix guardrails:
    # - do NOT buy YES below this floor (kills the 6–8c longshot behavior)
    # - do NOT buy NO below this floor (symmetric)
    YES_PRICE_FLOOR = 15.0
    NO_PRICE_FLOOR = 15.0

    # Prefer “higher probability” areas (you said you’re okay with ~$90 stakes if likely to win):
    # That means you generally want to be on the expensive side, not ultra-cheap tails.
    # We'll allow both sides, but penalize tail-like pricing hard.
    def score_side(side: str) -> Tuple[float, float, List[str]]:
        """
        Returns (score, edge_bps, side_reasons)
        """
        sr: List[str] = []

        if side == "YES":
            px = mid_yes
            if px < YES_PRICE_FLOOR:
                sr.append(f"yes_below_floor:{px:.1f}")
                return 0.0, 0.0, sr

            # Prefer moderately cheap YES, not tails
            s = base
            edge = 0.0

            if 15.0 <= px <= 35.0:
                s += 10
                edge = 55.0
                sr.append("yes_value_zone")
            elif 35.0 < px <= 45.0:
                s += 4
                edge = 35.0
                sr.append("yes_transition")
            elif 45.0 < px < 55.0:
                s -= 8
                edge = 10.0
                sr.append("coinflip_penalty")
            elif px >= 55.0:
                # Buying YES when already pricey tends to be worse than just buying NO
                s -= 10
                edge = 0.0
                sr.append("yes_too_pricey")

            return s, edge, sr

        # side == "NO"
        px = mid_no
        if px < NO_PRICE_FLOOR:
            sr.append(f"no_below_floor:{px:.1f}")
            return 0.0, 0.0, sr

        s = base
        edge = 0.0

        # Buying NO is equivalent to fading expensive YES.
        # Prefer when YES is expensive (i.e., NO mid is reasonably cheap but not a tail).
        if 15.0 <= px <= 35.0:
            s += 10
            edge = 55.0
            sr.append("no_value_zone")
        elif 35.0 < px <= 45.0:
            s += 4
            edge = 35.0
            sr.append("no_transition")
        elif 45.0 < px < 55.0:
            s -= 8
            edge = 10.0
            sr.append("coinflip_penalty")
        elif px >= 55.0:
            s -= 10
            edge = 0.0
            sr.append("no_too_pricey")

        return s, edge, sr

    yes_score, yes_edge, yes_r = score_side("YES")
    no_score, no_edge, no_r = score_side("NO")

    # Clamp and choose best
    yes_score = max(0.0, min(100.0, yes_score))
    no_score = max(0.0, min(100.0, no_score))

    # If both are bad, SKIP
    if yes_score < float(min_conviction) and no_score < float(min_conviction):
        # Keep the higher score for diagnostics even if skipping
        best = max(yes_score, no_score, 0.0)
        why = ["below_threshold:" + str(int(min_conviction))]
        # Attach both-side diagnostics
        why += [f"mid_yes:{mid_yes:.1f}", f"mid_no:{mid_no:.1f}"]
        if yes_r:
            why += ["yes:" + ",".join(yes_r)]
        if no_r:
            why += ["no:" + ",".join(no_r)]
        return SignalDecision("SKIP", best, 0.0, reasons + why)

    # Choose the best passing side
    if yes_score >= no_score:
        direction = "YES"
        score = yes_score
        edge_bps = yes_edge
        reasons_out = reasons + [f"mid_yes:{mid_yes:.1f}", f"mid_no:{mid_no:.1f}"] + ["yes:" + ",".join(yes_r)]
    else:
        direction = "NO"
        score = no_score
        edge_bps = no_edge
        reasons_out = reasons + [f"mid_yes:{mid_yes:.1f}", f"mid_no:{mid_no:.1f}"] + ["no:" + ",".join(no_r)]

    # Final clamp
    score = max(0.0, min(100.0, score))

    return SignalDecision(direction, score, edge_bps, reasons_out)