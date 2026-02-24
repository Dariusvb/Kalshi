from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioSnapshot:
    cash_balance_dollars: float = 0.0
    open_exposure_dollars: float = 0.0
    open_positions_count: int = 0
    daily_pnl_dollars: float = 0.0


def parse_balance_payload(payload: dict) -> float:
    # Best-effort normalization; adjust if payload field names differ
    # Common possibilities: balance, cash_balance, available_funds in cents
    if not isinstance(payload, dict):
        return 0.0
    for key in ("balance", "cash_balance", "available_funds", "cash"):
        if key in payload:
            val = payload[key]
            if isinstance(val, (int, float)):
                # If it's huge, assume cents
                return float(val / 100.0) if val > 10000 else float(val)
    return 0.0
