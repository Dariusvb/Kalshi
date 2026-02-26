from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from app.utils import gen_client_id


@dataclass
class ExecutionResult:
    action: str
    response: dict


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp_price_cents(x: float) -> int:
    try:
        return int(max(1, min(99, round(float(x)))))
    except Exception:
        return 50


def cents_to_contract_count(stake_dollars: float, price_cents: int) -> int:
    """
    Rough count sizing for binary contracts (BUY side).
    stake_dollars ~ max risk on BUY side at price cents.
    """
    if price_cents <= 0:
        return 0
    stake_cents = int(round(float(stake_dollars) * 100.0))
    return max(1, stake_cents // int(price_cents))


def _infer_no_quotes_from_yes(yes_bid: float, yes_ask: float) -> Tuple[Optional[float], Optional[float]]:
    """
    If explicit NO bid/ask aren't present, infer from YES quotes using complements:

      no_ask ≈ 100 - yes_bid
      no_bid ≈ 100 - yes_ask

    This is a common complement relationship for binary markets.
    """
    no_ask = None
    no_bid = None
    if yes_bid and yes_bid > 0:
        no_ask = 100.0 - float(yes_bid)
    if yes_ask and yes_ask > 0:
        no_bid = 100.0 - float(yes_ask)
    return no_bid, no_ask


def _best_prices_for_side(market: dict) -> dict:
    """
    Normalize and infer best available quotes for YES and NO sides.

    Returns:
      {
        "mid_yes": float,
        "yes_bid": float,
        "yes_ask": float,
        "no_bid": float,
        "no_ask": float,
      }
    """
    mid_yes = _to_float(market.get("mid_yes_cents"), 50.0)

    yes_bid = _to_float(market.get("yes_bid"), 0.0)
    yes_ask = _to_float(market.get("yes_ask"), 0.0)

    no_bid = _to_float(market.get("no_bid"), 0.0)
    no_ask = _to_float(market.get("no_ask"), 0.0)

    # Infer missing NO quotes from YES quotes if needed
    if (no_bid <= 0 and no_ask <= 0) and (yes_bid > 0 or yes_ask > 0):
        inf_no_bid, inf_no_ask = _infer_no_quotes_from_yes(yes_bid, yes_ask)
        if no_bid <= 0 and inf_no_bid is not None:
            no_bid = float(inf_no_bid)
        if no_ask <= 0 and inf_no_ask is not None:
            no_ask = float(inf_no_ask)

    # If YES quotes are missing but NO quotes exist, we can infer YES too
    # (symmetry: yes_ask ≈ 100 - no_bid, yes_bid ≈ 100 - no_ask)
    if (yes_bid <= 0 and yes_ask <= 0) and (no_bid > 0 or no_ask > 0):
        if yes_ask <= 0 and no_bid > 0:
            yes_ask = 100.0 - float(no_bid)
        if yes_bid <= 0 and no_ask > 0:
            yes_bid = 100.0 - float(no_ask)

    # If mid is missing/invalid, infer from best available two-sided YES quote or NO quote
    if mid_yes <= 0 or mid_yes >= 100:
        if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
            mid_yes = 0.5 * (yes_bid + yes_ask)
        elif no_bid > 0 and no_ask > 0 and no_ask >= no_bid:
            mid_yes = 100.0 - 0.5 * (no_bid + no_ask)
        else:
            mid_yes = 50.0

    return {
        "mid_yes": float(mid_yes),
        "yes_bid": float(yes_bid),
        "yes_ask": float(yes_ask),
        "no_bid": float(no_bid),
        "no_ask": float(no_ask),
    }


def build_limit_order_from_signal(ticker: str, direction: str, market: dict, stake_dollars: float) -> dict:
    """
    Build a BUY limit order for YES or NO that is compatible with main.py.

    Key behavior:
    - Returns yes_no as lowercase ("yes"/"no") to match main.py + KalshiClient usage.
    - Uses the best available side-specific ask for entry (buying crosses to ask).
    - Robust to missing no_bid/no_ask by inferring from yes_bid/yes_ask (and vice versa).
    - Safe fallback to mid when quotes are missing.
    """
    d = str(direction or "").strip().upper()
    if d not in {"YES", "NO"}:
        raise ValueError("direction must be YES or NO")

    q = _best_prices_for_side(market)
    mid_yes = q["mid_yes"]
    yes_bid = q["yes_bid"]
    yes_ask = q["yes_ask"]
    no_bid = q["no_bid"]
    no_ask = q["no_ask"]

    side = "buy"

    if d == "YES":
        # Prefer crossing to best displayed YES ask; fallback to mid; final fallback to (bid+1) if bid exists.
        if yes_ask > 0:
            price = _clamp_price_cents(yes_ask)
        elif mid_yes > 0:
            price = _clamp_price_cents(mid_yes)
        elif yes_bid > 0:
            price = _clamp_price_cents(yes_bid + 1.0)
        else:
            price = 50
        yes_no = "yes"

    else:  # d == "NO"
        # Prefer crossing to best displayed NO ask; fallback via complements; then fallback to inferred no-mid.
        if no_ask > 0:
            price = _clamp_price_cents(no_ask)
        elif yes_bid > 0:
            # no_ask ≈ 100 - yes_bid
            price = _clamp_price_cents(100.0 - yes_bid)
        elif mid_yes > 0:
            price = _clamp_price_cents(100.0 - mid_yes)
        elif no_bid > 0:
            price = _clamp_price_cents(no_bid + 1.0)
        else:
            price = 50
        yes_no = "no"

    count = cents_to_contract_count(float(stake_dollars), int(price))

    return {
        "ticker": str(ticker),
        "side": side,
        "yes_no": yes_no,  # IMPORTANT: lowercase for main.py / client.place_order
        "count": int(count),
        "price_cents": int(price),
        "client_order_id": gen_client_id("kalshi"),
    }