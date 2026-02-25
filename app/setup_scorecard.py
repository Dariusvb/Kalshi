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


def build_setup_scorecard(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Consumes decision rows (preferably recent 100-500).
    Uses resolved rows when available.
    """
    rows = [d for d in decisions if str(d.get("action", "")).startswith("TRADE")]

    by_category: dict[str, SetupBucketStats] = {}
    by_conviction: dict[str, SetupBucketStats] = {}
    by_social: dict[str, SetupBucketStats] = {"social_on": SetupBucketStats(), "social_off": SetupBucketStats()}
    by_news: dict[str, SetupBucketStats] = {"news_pos": SetupBucketStats(), "news_neutral": SetupBucketStats(), "news_neg": SetupBucketStats()}
    by_spread: dict[str, SetupBucketStats] = {}

    total = SetupBucketStats()

    for r in rows:
        conv = float(r.get("final_conviction", 0.0) or 0.0)
        cat = str(r.get("market_category") or "unknown").lower()
        spread = r.get("spread_cents")  # optional if/when you store it
        social_bonus = float(r.get("social_bonus", 0.0) or 0.0)
        news_score = r.get("news_score")

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

        # news
        try:
            ns = float(news_score) if news_score is not None else 0.0
        except Exception:
            ns = 0.0
        if ns > 1.0:
            apply(by_news["news_pos"])
        elif ns < -1.0:
            apply(by_news["news_neg"])
        else:
            apply(by_news["news_neutral"])

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

    return {
        "total": {
            "n": total.n,
            "resolved": total.resolved,
            "wins": total.wins,
            "losses": total.losses,
            "win_rate": round(total.win_rate * 100.0, 1) if total.win_rate is not None else None,
            "pnl": round(total.pnl, 2),
            "expectancy": round(total.expectancy, 4) if total.expectancy is not None else None,
        },
        "by_category": _serialize_map(by_category),
        "by_conviction": _serialize_map(by_conviction),
        "by_social": _serialize_map(by_social),
        "by_news": _serialize_map(by_news),
        "by_spread": _serialize_map(by_spread),
    }


def summarize_scorecard_for_discord(scorecard: dict[str, Any], top_n: int = 3) -> str:
    total = scorecard.get("total", {})
    parts = [
        "🧠 Setup Scorecard",
        f"Total trades={total.get('n', 0)} resolved={total.get('resolved', 0)} "
        f"W/L={total.get('wins', 0)}-{total.get('losses', 0)} "
        f"WR={total.get('win_rate', 'n/a')}% "
        f"PnL=${total.get('pnl', 0):.2f} "
        f"Exp={total.get('expectancy', 'n/a')}",
    ]

    # Top categories by pnl (with minimum sample)
    cats = scorecard.get("by_category", {})
    ranked = []
    for k, v in cats.items():
        if int(v.get("resolved") or 0) < 3:
            continue
        ranked.append((k, float(v.get("pnl") or 0.0), float(v.get("win_rate") or 0.0), int(v.get("resolved") or 0)))
    ranked.sort(key=lambda x: x[1], reverse=True)

    if ranked:
        best = ranked[:top_n]
        cat_line = "Top cats: " + " | ".join(
            [f"{k} PnL=${p:.2f} WR={wr:.1f}% n={n}" for (k, p, wr, n) in best]
        )
        parts.append(cat_line)

    convs = scorecard.get("by_conviction", {})
    # highlight strongest conviction bucket with >=3 resolved by expectancy
    conv_ranked = []
    for k, v in convs.items():
        if int(v.get("resolved") or 0) < 3:
            continue
        exp = v.get("expectancy")
        if exp is None:
            continue
        conv_ranked.append((k, float(exp), int(v.get("resolved") or 0)))
    conv_ranked.sort(key=lambda x: x[1], reverse=True)
    if conv_ranked:
        k, exp, n = conv_ranked[0]
        parts.append(f"Best conv bucket: {k} Exp=${exp:.4f} n={n}")

    return "\n".join(parts)
