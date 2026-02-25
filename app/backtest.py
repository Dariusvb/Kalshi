from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional


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


# =============================
# News tuning models / helpers
# =============================

@dataclass
class NewsTuningParams:
    news_min_confidence_to_apply: float = 0.35
    news_full_confidence_at: float = 0.75
    news_conflict_penalty_multiplier: float = 0.35
    news_neutral_regime_multiplier: float = 0.40
    news_mixed_regime_multiplier: float = 0.60
    news_abs_effect_cap_on_conviction: float = 6.0
    news_allow_direction_flip: bool = False


@dataclass
class NewsTuningResult:
    params: NewsTuningParams
    n_rows: int
    n_resolved: int
    n_classified: int
    n_conflicts: int
    avg_effective_news_abs: float
    score: float
    win_rate_top_bucket: Optional[float]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        p = self.params
        return {
            "score": self.score,
            "n_rows": self.n_rows,
            "n_resolved": self.n_resolved,
            "n_classified": self.n_classified,
            "n_conflicts": self.n_conflicts,
            "avg_effective_news_abs": self.avg_effective_news_abs,
            "win_rate_top_bucket": self.win_rate_top_bucket,
            "summary": self.summary,
            "params": {
                "news_min_confidence_to_apply": p.news_min_confidence_to_apply,
                "news_full_confidence_at": p.news_full_confidence_at,
                "news_conflict_penalty_multiplier": p.news_conflict_penalty_multiplier,
                "news_neutral_regime_multiplier": p.news_neutral_regime_multiplier,
                "news_mixed_regime_multiplier": p.news_mixed_regime_multiplier,
                "news_abs_effect_cap_on_conviction": p.news_abs_effect_cap_on_conviction,
                "news_allow_direction_flip": p.news_allow_direction_flip,
            },
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
        "news_score", "news_confidence", "news_regime", "news_effective_score", "spread_cents",
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


# =============================
# News tuning internals
# =============================

def _parse_reason_kv(reasons: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not reasons:
        return out
    try:
        for part in str(reasons).split(";"):
            p = part.strip()
            if not p or ":" not in p:
                continue
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return out
    return out


def _extract_row_features_for_tuning(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not _looks_trade(row):
        return None
    if row.get("resolved_ts") is None:
        return None

    won = _won_value(row)
    if won is None:
        return None

    reasons_map = _parse_reason_kv(row.get("reasons", ""))

    base_conv = _safe_float(row.get("base_conviction")) or 0.0
    social_bonus = _safe_float(row.get("social_bonus")) or 0.0
    final_conv = _safe_float(row.get("final_conviction")) or 0.0

    raw_news = _safe_float(row.get("news_score"))
    if raw_news is None:
        raw_news = _safe_float(reasons_map.get("news_score_raw"))
    if raw_news is None:
        raw_news = 0.0

    news_conf = _safe_float(row.get("news_confidence"))
    if news_conf is None:
        news_conf = _safe_float(reasons_map.get("news_conf"))
    if news_conf is None:
        news_conf = 0.0

    news_regime = str(row.get("news_regime") or reasons_map.get("news") or "unavailable")
    direction = str(row.get("direction", "SKIP")).upper()

    return {
        "id": row.get("id"),
        "direction": direction,
        "won": won,
        "base_conviction": float(base_conv),
        "social_bonus": float(social_bonus),
        "final_conviction": float(final_conv),
        "raw_news_score": float(raw_news),
        "news_confidence": float(news_conf),
        "news_regime": news_regime,
        "realized_pnl": _safe_float(row.get("realized_pnl")),
    }


def _apply_news_effect_candidate(
    *,
    base_signal_direction: str,
    raw_news_score: float,
    news_confidence: float,
    news_regime: str,
    params: NewsTuningParams,
) -> tuple[float, bool]:
    raw_news_score = float(raw_news_score)
    news_conf = max(0.0, min(1.0, float(news_confidence)))
    news_regime = str(news_regime or "unavailable")
    sig_dir = str(base_signal_direction or "SKIP").upper()

    if news_conf <= params.news_min_confidence_to_apply:
        conf_scale = 0.10
    elif news_conf >= params.news_full_confidence_at:
        conf_scale = 1.00
    else:
        span = max(1e-9, params.news_full_confidence_at - params.news_min_confidence_to_apply)
        conf_scale = 0.10 + 0.90 * ((news_conf - params.news_min_confidence_to_apply) / span)

    regime_scale = 1.0
    if news_regime in {"noisy_or_weak", "unavailable", "disabled", "error_fallback"}:
        regime_scale = params.news_neutral_regime_multiplier
    elif news_regime in {"mixed"}:
        regime_scale = params.news_mixed_regime_multiplier

    conflict = False
    if sig_dir == "YES" and raw_news_score < 0:
        conflict = True
    elif sig_dir == "NO" and raw_news_score > 0:
        conflict = True

    direction_scale = params.news_conflict_penalty_multiplier if conflict else 1.0

    eff = raw_news_score * conf_scale * regime_scale * direction_scale

    if (not params.news_allow_direction_flip) and conflict:
        eff = max(-2.0, min(2.0, eff))

    eff = max(-params.news_abs_effect_cap_on_conviction, min(params.news_abs_effect_cap_on_conviction, eff))
    return float(eff), conflict


def _evaluate_news_param_set(rows: list[dict[str, Any]], params: NewsTuningParams) -> NewsTuningResult:
    feats = []
    n_conflicts = 0
    eff_abs_vals: list[float] = []

    for r in rows:
        fx = _extract_row_features_for_tuning(r)
        if not fx:
            continue

        eff_news, conflict = _apply_news_effect_candidate(
            base_signal_direction=fx["direction"],
            raw_news_score=fx["raw_news_score"],
            news_confidence=fx["news_confidence"],
            news_regime=fx["news_regime"],
            params=params,
        )

        rebuilt_final = max(0.0, min(100.0, fx["base_conviction"] + fx["social_bonus"] + eff_news))

        fx["effective_news_score_candidate"] = eff_news
        fx["rebuilt_final_candidate"] = rebuilt_final
        fx["conflict_candidate"] = conflict

        feats.append(fx)
        eff_abs_vals.append(abs(eff_news))
        if conflict:
            n_conflicts += 1

    if not feats:
        return NewsTuningResult(
            params=params,
            n_rows=len(rows),
            n_resolved=0,
            n_classified=0,
            n_conflicts=0,
            avg_effective_news_abs=0.0,
            score=-9999.0,
            win_rate_top_bucket=None,
            summary="No usable resolved trades with won/loss labels.",
        )

    n_classified = len(feats)
    overall_wins = sum(1 for x in feats if x["won"] == 1)
    overall_wr = overall_wins / n_classified if n_classified else 0.0

    feats_sorted = sorted(feats, key=lambda x: x["rebuilt_final_candidate"], reverse=True)
    top_k = max(3, int(round(len(feats_sorted) * 0.30)))
    top = feats_sorted[:top_k]
    top_wr = (sum(1 for x in top if x["won"] == 1) / len(top)) if top else None

    strong_news = [x for x in feats if abs(x["effective_news_score_candidate"]) >= 1.0]
    strong_news_wr = (
        sum(1 for x in strong_news if x["won"] == 1) / len(strong_news)
        if strong_news else None
    )

    pnl_vals = [float(x["realized_pnl"]) for x in feats if x.get("realized_pnl") is not None]
    avg_pnl = (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else 0.0

    avg_eff_abs = (sum(eff_abs_vals) / len(eff_abs_vals)) if eff_abs_vals else 0.0

    wr_lift = ((top_wr - overall_wr) if top_wr is not None else 0.0)
    strong_news_bonus = ((strong_news_wr - overall_wr) if strong_news_wr is not None else 0.0)
    overreach_penalty = max(0.0, avg_eff_abs - 2.5) * 0.08

    score = (
        (wr_lift * 100.0) +
        (strong_news_bonus * 35.0) +
        (avg_pnl * 0.05) -
        overreach_penalty
    )

    summary = (
        f"rows={len(rows)} usable={n_classified} overall_wr={overall_wr*100:.1f}% "
        f"top30_wr={(top_wr*100 if top_wr is not None else float('nan')):.1f}% "
        f"wr_lift={wr_lift*100:.1f}pts "
        f"strong_news_n={len(strong_news)} "
        f"avg|news_eff|={avg_eff_abs:.2f} conflicts={n_conflicts} "
        f"avg_pnl={avg_pnl:.4f}"
    )

    return NewsTuningResult(
        params=params,
        n_rows=len(rows),
        n_resolved=len(feats),
        n_classified=n_classified,
        n_conflicts=n_conflicts,
        avg_effective_news_abs=round(avg_eff_abs, 4),
        score=round(score, 4),
        win_rate_top_bucket=round(top_wr * 100.0, 1) if top_wr is not None else None,
        summary=summary,
    )


def tune_news_parameters_from_decisions(
    decisions: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[NewsTuningResult]:
    min_conf_grid = [0.25, 0.35, 0.45]
    full_conf_grid = [0.70, 0.75, 0.85]
    conflict_mult_grid = [0.20, 0.35, 0.50]
    neutral_regime_grid = [0.25, 0.40, 0.55]
    mixed_regime_grid = [0.45, 0.60, 0.75]
    effect_cap_grid = [4.0, 6.0, 8.0]

    results: list[NewsTuningResult] = []

    for (
        min_conf,
        full_conf,
        conflict_mult,
        neutral_mult,
        mixed_mult,
        effect_cap,
    ) in product(
        min_conf_grid,
        full_conf_grid,
        conflict_mult_grid,
        neutral_regime_grid,
        mixed_regime_grid,
        effect_cap_grid,
    ):
        if full_conf <= min_conf:
            continue

        params = NewsTuningParams(
            news_min_confidence_to_apply=min_conf,
            news_full_confidence_at=full_conf,
            news_conflict_penalty_multiplier=conflict_mult,
            news_neutral_regime_multiplier=neutral_mult,
            news_mixed_regime_multiplier=mixed_mult,
            news_abs_effect_cap_on_conviction=effect_cap,
            news_allow_direction_flip=False,
        )
        results.append(_evaluate_news_param_set(decisions, params))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:max(1, int(top_k))]


def _print_news_tuning_results(results: list[NewsTuningResult]) -> None:
    if not results:
        print("No tuning results.")
        return

    print("\n=== News Parameter Tuning (Top Results) ===")
    for i, r in enumerate(results, start=1):
        p = r.params
        print(
            f"{i}. score={r.score:.3f} | usable={r.n_classified} | "
            f"top30_wr={r.win_rate_top_bucket}% | avg|eff|={r.avg_effective_news_abs:.2f}"
        )
        print(
            "   params: "
            f"min_conf={p.news_min_confidence_to_apply:.2f}, "
            f"full_conf={p.news_full_confidence_at:.2f}, "
            f"conflict_mult={p.news_conflict_penalty_multiplier:.2f}, "
            f"neutral_mult={p.news_neutral_regime_multiplier:.2f}, "
            f"mixed_mult={p.news_mixed_regime_multiplier:.2f}, "
            f"cap={p.news_abs_effect_cap_on_conviction:.1f}"
        )
        print(f"   {r.summary}")


def run_news_tuning_report(
    db_path: str,
    limit: int = 1000,
    mode: Optional[str] = None,
    top_k: int = 5,
    json_out: bool = False,
) -> Dict[str, Any]:
    rows = load_decisions(db_path=db_path, limit=limit, only_mode=mode, only_trades=False)

    if not rows:
        result = {"ok": True, "message": "No decisions found.", "rows": 0, "tuning": []}
        if json_out:
            print(json.dumps(result, indent=2))
        else:
            print("No decisions found.")
        return result

    results = tune_news_parameters_from_decisions(rows, top_k=top_k)
    _print_news_tuning_results(results)

    payload = {
        "ok": True,
        "rows": len(rows),
        "mode": mode,
        "top_k": top_k,
        "tuning": [r.to_dict() for r in results],
    }

    if json_out:
        print("\n=== JSON TUNING REPORT ===")
        print(json.dumps(payload, indent=2))

    return payload


# =============================
# Existing report mode
# =============================

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

    # New: optional news tuning mode
    p.add_argument("--tune-news", action="store_true", help="Run news parameter tuning grid search instead of standard report")
    p.add_argument("--tune-top-k", type=int, default=5, help="Top N parameter sets to show for --tune-news (default: 5)")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if bool(args.tune_news):
        run_news_tuning_report(
            db_path=args.db_path,
            limit=args.limit,
            mode=args.mode,
            top_k=int(args.tune_top_k),
            json_out=bool(args.json),
        )
        return

    run_backtest_report(
        db_path=args.db_path,
        limit=args.limit,
        mode=args.mode,
        json_out=bool(args.json),
    )


if __name__ == "__main__":
    main()
