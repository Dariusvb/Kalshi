from __future__ import annotations

from typing import List, Dict


def summarize_decisions(decisions: List[dict]) -> dict:
    if not decisions:
        return {
            "count": 0,
            "avg_conviction": 0.0,
            "avg_stake": 0.0,
            "trades": 0,
            "skips": 0,
        }
    count = len(decisions)
    avg_conv = sum(d["final_conviction"] for d in decisions) / count
    avg_stake = sum(d["stake_dollars"] for d in decisions) / count
    trades = sum(1 for d in decisions if d["action"].startswith("TRADE"))
    skips = sum(1 for d in decisions if d["action"] == "SKIP")
    return {
        "count": count,
        "avg_conviction": round(avg_conv, 2),
        "avg_stake": round(avg_stake, 2),
        "trades": trades,
        "skips": skips,
    }
