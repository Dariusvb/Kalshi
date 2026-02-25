from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SetupBucketStats:
    n: int = 0
    resolved: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        denom = self.wins + self.losses
        if denom == 0:
            return None
        return self.wins / denom

    @property
    def expectancy(self) -> Optional[float]:
        if self.resolved == 0:
            return None
        return self.pnl / self.resolved


def _conv_bucket(conv: float) -> str:
    if conv < 55:
        return "50-54"
    if conv < 60:
        return "55-59"
    if conv < 65:
        return "60-64"
    if conv < 70:
        return "65-69"
    if conv < 75:
        return "70-74"
    if conv < 80:
        return "75-79"
    return "80+"


def _spread_bucket(spread: Optional[float]) -> str:
    if spread is None:
        return "unknown"
    s = float(spread)
    if s <= 1:
        return "0-1"
    if s <= 3:
        return "2-3"
    if s <= 5:
        return "4-5"
    return "6+"


def _news_score_bucket(score: Optional[float]) -> str:
    """
    Buckets for either raw news_score or effective applied news score.
    Tuned for your bounded news impact design.
    """
    if score is None:
        return "unknown"
    s = float(score)

    if s <= -4:
        return "<=-4"
    if s <= -2:
        return "-4..-2"
    if s < -0.5:
        return "-2..-0.5"
    if s <= 0.5:
        return "-0.5..0.5"
    if s < 2:
        return "0.5..2"
    if s < 4:
        return "2..4"
    return ">=4"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _parse_reason_kv(reasons: Any) -> dict[str, str]:
    """
    Parses semicolon-delimited reasons strings like:
      "...;news_score_eff:1.23;news_conflict_sig:1;..."
    """
    out: dict[str, str] = {}
    if not reasons:
        return out
    try:
        parts = str(reasons).split(";")
        for part in parts:
            p = part.strip()
            if not p or ":" not in p:
                continue
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return out
    return out


def build_setup_scorecard(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Consumes decision rows (preferably recent 100-500).
    Uses resolved rows when available.

    Supports both older and newer schemas:
      - news_score (raw)
      - news_effective_score (preferred, if stored)
      - news_regime
      - reasons fallback parsing for:
          news_score_eff, news_conflict_sig, etc.
    """
    rows = [d for d in decisions if str(d.get("action", "")).startswith("TRADE")]

    by_category: dict[str, SetupBucketStats] = {}
    by_conviction: dict[str, SetupBucketStats] = {}
    by_social: dict[str, SetupBucketStats] = {
        "social_on": SetupBucketStats(),
        "social_off": SetupBucketStats(),
    }

    # Legacy/simple news buckets (raw-score based)
    by_news: dict[str, SetupBucketStats] = {
        "news_pos": SetupBucketStats(),
        "news_neutral": SetupBucketStats(),
        "news_neg": SetupBucketStats(),
    }

    # New richer news diagnostics
    by_news_regime: dict[str, SetupBucketStats] = {}
    by_news_raw: dict[str, SetupBucketStats] = {}
    by_news_effective: dict[str, SetupBucketStats] = {}
    by_news_conflict_signal: dict[str, SetupBucketStats] = {
        "aligned_or_unknown": SetupBucketStats(),
        "conflicts_signal": SetupBucketStats(),
    }

    by_spread: dict[str, SetupBucketStats] = {}
    total = SetupBucketStats()

    for r in rows:
        conv = float(r.get("final_conviction", 0.0) or 0.0)
        cat = str(r.get("market_category") or "unknown").lower()
        spread = r.get("spread_cents")
        social_bonus = float(r.get("social_bonus", 0.0) or 0.0)

        # News fields (new schema + fallback parsing from reasons)
        reasons_map = _parse_reason_kv(r.get("reasons", ""))
        news_regime = str(r.get("news_regime") or reasons_map.get("news") or "unknown").strip().lower() or "unknown"

        raw_news_score = _safe_float(r.get("news_score"), None)
        if raw_news_score is None:
            raw_news_score = _safe_float(reasons_map.get("news_score_raw"), 0.0)

        eff_news_score = _safe_float(r.get("news_effective_score"), None)
        if eff_news_score is None:
            eff_news_score = _safe_float(reasons_map.get("news_score_eff"), 0.0)

        # conflict flag from reasons (new main.py appends news_conflict_sig:0/1)
        news_conflict_sig = False
        raw_conflict = reasons_map.get("news_conflict_sig")
        if raw_conflict is not None:
            news_conflict_sig = str(raw_conflict).strip().lower() in {"1", "true", "yes"}

        won_val = r.get("won")
        pnl_val = r.get("realized_pnl")
        resolved = r.get("resolved_ts") is not None

        def apply(bucket: SetupBucketStats) -> None:
            bucket.n += 1
            if resolved:
                bucket.resolved += 1
            if won_val is True or won_val == 1:
                bucket.wins += 1
            elif won_val is False or won_val == 0:
                bucket.losses += 1
            if pnl_val is not None:
                try:
                    bucket.pnl += float(pnl_val)
                except Exception:
                    pass

        # total
        apply(total)

        # category
        by_category.setdefault(cat, SetupBucketStats())
        apply(by_category[cat])

        # conviction
        cb = _conv_bucket(conv)
        by_conviction.setdefault(cb, SetupBucketStats())
        apply(by_conviction[cb])

        # social
        apply(by_social["social_on" if social_bonus > 0 else "social_off"])

        # legacy/simple news classification (raw score)
        ns = float(raw_news_score or 0.0)
        if ns > 1.0:
            apply(by_news["news_pos"])
        elif ns < -1.0:
            apply(by_news["news_neg"])
        else:
            apply(by_news["news_neutral"])

        # news regime
        by_news_regime.setdefault(news_regime, SetupBucketStats())
        apply(by_news_regime[news_regime])

        # raw news bucket
        raw_bucket = _news_score_bucket(raw_news_score)
        by_news_raw.setdefault(raw_bucket, SetupBucketStats())
        apply(by_news_raw[raw_bucket])

        # effective/applied news bucket
        eff_bucket = _news_score_bucket(eff_news_score)
        by_news_effective.setdefault(eff_bucket, SetupBucketStats())
        apply(by_news_effective[eff_bucket])

        # conflict vs aligned/unknown
        apply(by_news_conflict_signal["conflicts_signal" if news_conflict_sig else "aligned_or_unknown"])

        # spread
        sb = _spread_bucket(spread if spread is not None else None)
        by_spread.setdefault(sb, SetupBucketStats())
        apply(by_spread[sb])

    def _serialize_map(m: dict[str, SetupBucketStats]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k, v in m.items():
            out[k] = {
                "n": v.n,
                "resolved": v.resolved,
                "wins": v.wins,
                "losses": v.losses,
                "win_rate": round(v.win_rate * 100.0, 1) if v.win_rate is not None else None,
                "pnl": round(v.pnl, 2),
                "expectancy": round(v.expectancy, 4) if v.expectancy is not None else None,
            }
        return out

    total_out = {
        "n": total.n,
        "resolved": total.resolved,
        "wins": total.wins,
        "losses": total.losses,
        "win_rate": round(total.win_rate * 100.0, 1) if total.win_rate is not None else None,
        "pnl": round(total.pnl, 2),
        "expectancy": round(total.expectancy, 4) if total.expectancy is not None else None,
    }

    return {
        "total": total_out,
        "by_category": _serialize_map(by_category),
        "by_conviction": _serialize_map(by_conviction),
        "by_social": _serialize_map(by_social),
        "by_news": _serialize_map(by_news),  # legacy/simple view (raw-based)
        "by_news_regime": _serialize_map(by_news_regime),
        "by_news_raw": _serialize_map(by_news_raw),
        "by_news_effective": _serialize_map(by_news_effective),
        "by_news_conflict_signal": _serialize_map(by_news_conflict_signal),
        "by_spread": _serialize_map(by_spread),
    }


def summarize_scorecard_for_discord(scorecard: dict[str, Any], top_n: int = 3) -> str:
    total = scorecard.get("total", {})

    total_pnl = float(total.get("pnl") or 0.0)
    total_exp = total.get("expectancy")
    total_exp_str = f"{float(total_exp):.4f}" if total_exp is not None else "n/a"

    parts = [
        "🧠 Setup Scorecard",
        f"Total trades={int(total.get('n', 0) or 0)} resolved={int(total.get('resolved', 0) or 0)} "
        f"W/L={int(total.get('wins', 0) or 0)}-{int(total.get('losses', 0) or 0)} "
        f"WR={total.get('win_rate', 'n/a')}% "
        f"PnL=${total_pnl:.2f} "
        f"Exp={total_exp_str}",
    ]

    def _top_by_metric(
        section: dict[str, Any],
        *,
        metric: str,
        min_resolved: int = 3,
        reverse: bool = True,
        limit: int = 3,
    ) -> list[tuple[str, float, int, Optional[float]]]:
        ranked: list[tuple[str, float, int, Optional[float]]] = []
        for k, v in section.items():
            resolved = int(v.get("resolved") or 0)
            if resolved < min_resolved:
                continue
            mv = v.get(metric)
            if mv is None:
                continue
            wr = v.get("win_rate")
            try:
                ranked.append((k, float(mv), resolved, float(wr) if wr is not None else None))
            except Exception:
                continue
        ranked.sort(key=lambda x: x[1], reverse=reverse)
        return ranked[:limit]

    # Top categories by pnl
    cats = scorecard.get("by_category", {})
    top_cats = _top_by_metric(cats, metric="pnl", min_resolved=3, reverse=True, limit=top_n)
    if top_cats:
        parts.append(
            "Top cats: " + " | ".join(
                [f"{k} PnL=${p:.2f} WR={(wr if wr is not None else 0.0):.1f}% n={n}" for (k, p, n, wr) in top_cats]
            )
        )

    # Best conviction bucket by expectancy
    convs = scorecard.get("by_conviction", {})
    best_conv = _top_by_metric(convs, metric="expectancy", min_resolved=3, reverse=True, limit=1)
    if best_conv:
        k, exp, n, _wr = best_conv[0]
        parts.append(f"Best conv bucket: {k} Exp=${exp:.4f} n={n}")

    # News regime insight
    regimes = scorecard.get("by_news_regime", {})
    # Prefer confirmed / mixed / noisy names if present, but rank by pnl among >=3 resolved
    top_regimes = _top_by_metric(regimes, metric="pnl", min_resolved=3, reverse=True, limit=2)
    if top_regimes:
        parts.append(
            "News regimes: " + " | ".join(
                [f"{k} PnL=${p:.2f} WR={(wr if wr is not None else 0.0):.1f}% n={n}" for (k, p, n, wr) in top_regimes]
            )
        )

    # Effective news bucket insight (what actually moved conviction)
    eff_news = scorecard.get("by_news_effective", {})
    top_eff = _top_by_metric(eff_news, metric="expectancy", min_resolved=3, reverse=True, limit=2)
    if top_eff:
        parts.append(
            "News eff buckets: " + " | ".join(
                [f"{k} Exp=${exp:.4f} n={n}" for (k, exp, n, _wr) in top_eff]
            )
        )

    # Conflict vs aligned/unknown comparison
    conflict_sec = scorecard.get("by_news_conflict_signal", {})
    aligned = conflict_sec.get("aligned_or_unknown", {})
    conflicted = conflict_sec.get("conflicts_signal", {})

    aligned_res = int(aligned.get("resolved") or 0)
    conflicted_res = int(conflicted.get("resolved") or 0)

    if aligned_res >= 3 or conflicted_res >= 3:
        a_wr = aligned.get("win_rate")
        c_wr = conflicted.get("win_rate")
        a_exp = aligned.get("expectancy")
        c_exp = conflicted.get("expectancy")

        def _fmt_num(x: Any, places: int) -> str:
            if x is None:
                return "n/a"
            try:
                return f"{float(x):.{places}f}"
            except Exception:
                return "n/a"

        parts.append(
            "News conflict check: "
            f"aligned WR={_fmt_num(a_wr,1)}% Exp=${_fmt_num(a_exp,4)} n={aligned_res} | "
            f"conflict WR={_fmt_num(c_wr,1)}% Exp=${_fmt_num(c_exp,4)} n={conflicted_res}"
        )

    return "\n".join(parts)
