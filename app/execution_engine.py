from __future__ import annotations

from dataclasses import dataclass

from app.utils import gen_client_id


@dataclass
class ExecutionResult:
    action: str
    response: dict


def cents_to_contract_count(stake_dollars: float, price_cents: int) -> int:
    """
    Rough count sizing for binary contracts.
    stake_dollars ~ max risk on BUY side at price cents.
    """
    if price_cents <= 0:
        return 0
    stake_cents = int(round(stake_dollars * 100))
    return max(1, stake_cents // price_cents)


def build_limit_order_from_signal(ticker: str, direction: str, market: dict, stake_dollars: float) -> dict:
    mid = int(round(market.get("mid_yes_cents", 50)))
    bid = int(market.get("yes_bid", max(1, mid - 1)))
    ask = int(market.get("yes_ask", min(99, mid + 1)))

    # Slightly aggressive but not reckless: cross to best displayed quote
    if direction == "YES":
        price = max(1, min(99, ask if ask > 0 else mid))
        yes_no = "YES"
        side = "buy"
    elif direction == "NO":
        # For NO exposure, many APIs still use yes/no + side semantics.
        # Here we model "buy NO" via yes_no=NO.
        # If your exact endpoint expects another schema, map in kalshi_client.place_order.
        price = max(1, min(99, 100 - (bid if bid > 0 else mid)))
        yes_no = "NO"
        side = "buy"
    else:
        raise ValueError("direction must be YES or NO")

    count = cents_to_contract_count(stake_dollars, price)
    return {
        "ticker": ticker,
        "side": side,
        "yes_no": yes_no,
        "count": count,
        "price_cents": price,
        "client_order_id": gen_client_id("kalshi"),
    }
