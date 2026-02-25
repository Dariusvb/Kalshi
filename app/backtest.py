from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BacktestStats:
    label: str
    count: int
    trades: int
    resolved_trades: int
    wins: int
    losses: int
    win_rate: Optional[float]
    pnl_total: Optional[float]
    expectancy: Optional[float]
    avg_stake: float
    avg_conviction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "trades": self.trades,
            "resolved_trades": self.resolved_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "pnl_total": self.pnl_total,
            "expectancy": self.expectancy,
            "avg_stake": self.avg_stake,
            "avg_conviction": self.avg_conviction,
        }


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return {str(r[1]) for r in rows}


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _looks_trade(row: Dict[str, Any]) -> bool:
    return str(row.get("action", "")).startswith("TRADE")


def _is_resolved_trade(row: Dict[str, Any]) -> bool:
    if not _looks_trade(row):
        return False
    return row.get("resolved_ts") is not None


def _won_value(row: Dict[str, Any]) -> Optional[int]:
    w = row.get("won")
    if w is None:
        return None
    if w in (0, 1):
        return int(w)
    if isinstance(w, bool):
        return 1 if w else 0
    if isinstance(w, str):
        s = w.strip().lower()
        if s in {"1", "true", "won", "win", "yes"}:
            return 1
        if s in {"0", "false", "lost", "loss", "no"}:
            return 0
    return None


def _contains_social_signal(row: Dict[str, Any]) -> bool:
    """
    Heuristic:
    - checks reasons for social tags
    - treats positive social_bonus as social-influenced
    """
    reasons = str(row.get("reasons", "") or "").lower()
    social_bonus = _safe_float(row.get("social_bonus")) or 0.0

    if social_bonus > 0:
        return True
    if "social:" in reasons and "disabled" not in reasons:
        return True
    return False


def _conviction_bucket(conv: Optional[float]) -> str:
    if conv is None:
        return "unknown"
    if conv < 55:
        return "<55"
    if conv < 65:
        return "55-64.9"
    if conv < 75:
        return "65-74.9"
    if conv < 85:
        return "75-84.9"
    return "85+"


def _spread_bucket(spread: Optional[int]) -> str:
    if spread is None:
        return "unknown"
    if spread <= 2:
        return "0-2"
    if spread <= 5:
        return "3-5"
    if spread <= 10:
        return "6-10"
    return "11+"


def _summarize_rows(rows: List[Dict[str, Any]], label: str) -> BacktestStats:
    count = len(rows)
    trades = [r for r in rows if _looks_trade(r)]
    resolved = [r for r in trades if _is_resolved_trade(r)]

    wins = 0
    losses = 0
    pnl_values: List[float] = []

    for r in resolved:
        w = _won_value(r)
        if w == 1:
            wins += 1
        elif w == 0:
            losses += 1

        pnl = _safe_float(r.get("realized_pnl"))
        if pnl is not None:
            pnl_values.append(pnl)

    win_rate = None
    if (wins + losses) > 0:
        win_rate = round((wins / (wins + losses)) * 100.0, 1)

    pnl_total = round(sum(pnl_values), 2) if pnl_values else None
    expectancy = round(sum(pnl_values) / len(pnl_values), 4) if pnl_values else None

    avg_stake = 0.0
    if trades:
        avg_stake = round(
            sum((_safe_float(r.get("stake_dollars")) or 0.0) for r in trades) / len(trades),
            2,
        )

    avg_conviction = 0.0
    if rows:
        avg_conviction = round(
            sum((_safe_float(r.get("final_conviction")) or 0.0) for r in rows) / len(rows),
            2,
        )

    return BacktestStats(
        label=label,
        count=count,
        trades=len(trades),
        resolved_trades=len(resolved),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        pnl_total=pnl_total,
        expectancy=expectancy,
        avg_stake=avg_stake,
        avg_conviction=avg_conviction,
    )


def _print_stats_table(title: str, stats_map: Dict[str, BacktestStats]) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Group':<24} {'Rows':>6} {'Trades':>7} {'Resolved':>9} "
        f"{'W':>4} {'L':>4} {'WinRate%':>9} {'PnL':>12} {'Exp':>10} "
        f"{'AvgStake':>10} {'AvgConv':>9}"
    )
    print(header)
    print("-" * len(header))

    for key, st in stats_map.items():
        win_rate = f"{st.win_rate:.1f}" if st.win_rate is not None else "n/a"
        pnl = f"{st.pnl_total:.2f}" if st.pnl_total is not None else "n/a"
        exp = f"{st.expectancy:.4f}" if st.expectancy is not None else "n/a"
        print(
            f"{key:<24} {st.count:>6} {st.trades:>7} {st.resolved_trades:>9} "
            f"{st.wins:>4} {st.losses:>4} {win_rate:>9} {pnl:>12} {exp:>10} "
            f"{st.avg_stake:>10.2f} {st.avg_conviction:>9.2f}"
        )


def _group_and_summarize(
    rows: List[Dict[str, Any]],
    key_fn,
    title: str,
    top_n: Optional[int] = None,
    min_rows: int = 1,
) -> Dict[str, BacktestStats]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(key_fn(r))
        groups.setdefault(k, []).append(r)

    stats_map: Dict[str, BacktestStats] = {}
    for k, sub in groups.items():
        if len(sub) < min_rows:
            continue
        stats_map[k] = _summarize_rows(sub, label=k)

    # Sort primarily by resolved trades then row count
    ordered_items = sorted(
        stats_map.items(),
        key=lambda kv: (kv[1].resolved_trades, kv[1].count),
        reverse=True,
    )
    if top_n is not None:
        ordered_items = ordered_items[:top_n]

    out = dict(ordered_items)
    _print_stats_table(title, out)
    return out


def load_decisions(
    db_path: str,
    limit: int = 1000,
    only_mode: Optional[str] = None,   # "sim" | "live" | None
    only_trades: bool = False,
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cols = _table_columns(conn, "decisions")
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Could not read decisions table from {db_path}: {e}") from e

    select_cols = [
        "id", "ts", "ticker", "direction", "base_conviction", "social_bonus",
        "final_conviction", "stake_dollars", "action", "reasons", "dry_run"
    ]
    optional_cols = [
        "realized_pnl", "won", "resolved_ts", "market_category",
        "news_score", "news_confidence", "news_regime", "spread_cents",
    ]
    for c in optional_cols:
        if c in cols:
            select_cols.append(c)

    sql = f"SELECT {', '.join(select_cols)} FROM decisions"
    where = []
    params: List[Any] = []

    if only_mode == "sim":
        where.append("dry_run = 1")
    elif only_mode == "live":
        where.append("dry_run = 0")

    if only_trades:
        where.append("action LIKE 'TRADE_%'")

    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def run_backtest_report(
    db_path: str,
    limit: int = 1000,
    mode: Optional[str] = None,
    json_out: bool = False,
) -> Dict[str, Any]:
    rows = load_decisions(db_path=db_path, limit=limit, only_mode=mode, only_trades=False)

    if not rows:
        result = {"ok": True, "message": "No decisions found.", "rows": 0}
        if json_out:
            print(json.dumps(result, indent=2))
        else:
            print("No decisions found.")
        return result

    overall = _summarize_rows(rows, "overall")
    sim_rows = [r for r in rows if bool(r.get("dry_run", True))]
    live_rows = [r for r in rows if not bool(r.get("dry_run", True))]

    sim_stats = _summarize_rows(sim_rows, "sim")
    live_stats = _summarize_rows(live_rows, "live")

    # Group comparisons
    by_mode = {
        "sim": sim_stats,
        "live": live_stats,
    }

    by_social = {
        "social_influenced": _summarize_rows([r for r in rows if _contains_social_signal(r)], "social_influenced"),
        "no_social": _summarize_rows([r for r in rows if not _contains_social_signal(r)], "no_social"),
    }

    by_news_regime = _group_and_summarize(
        rows,
        key_fn=lambda r: r.get("news_regime") or "unknown",
        title="By News Regime",
        top_n=10,
        min_rows=1,
    )

    by_category = _group_and_summarize(
        rows,
        key_fn=lambda r: r.get("market_category") or "unknown",
        title="By Market Category",
        top_n=15,
        min_rows=1,
    )

    by_conviction_bucket = _group_and_summarize(
        rows,
        key_fn=lambda r: _conviction_bucket(_safe_float(r.get("final_conviction"))),
        title="By Conviction Bucket",
        top_n=None,
        min_rows=1,
    )

    by_spread_bucket = _group_and_summarize(
        rows,
        key_fn=lambda r: _spread_bucket(_safe_int(r.get("spread_cents"))),
        title="By Spread Bucket",
        top_n=None,
        min_rows=1,
    )

    # Print main sections
    _print_stats_table("Overall", {"overall": overall})
    _print_stats_table("SIM vs LIVE", by_mode)
    _print_stats_table("Social vs No Social", by_social)

    result = {
        "ok": True,
        "rows": len(rows),
        "overall": overall.to_dict(),
        "by_mode": {k: v.to_dict() for k, v in by_mode.items()},
        "by_social": {k: v.to_dict() for k, v in by_social.items()},
        "by_news_regime": {k: v.to_dict() for k, v in by_news_regime.items()},
        "by_category": {k: v.to_dict() for k, v in by_category.items()},
        "by_conviction_bucket": {k: v.to_dict() for k, v in by_conviction_bucket.items()},
        "by_spread_bucket": {k: v.to_dict() for k, v in by_spread_bucket.items()},
    }

    if json_out:
        print("\n=== JSON REPORT ===")
        print(json.dumps(result, indent=2))

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kalshi bot replay/backtest report from SQLite decisions table.")
    p.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB (e.g. data/kalshi_bot.db)")
    p.add_argument("--limit", type=int, default=1000, help="Max recent decisions to analyze (default: 1000)")
    p.add_argument("--mode", choices=["sim", "live"], default=None, help="Analyze only sim or live mode")
    p.add_argument("--json", action="store_true", help="Also print JSON output")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_backtest_report(
        db_path=args.db_path,
        limit=args.limit,
        mode=args.mode,
        json_out=bool(args.json),
    )


if __name__ == "__main__":
    main()
