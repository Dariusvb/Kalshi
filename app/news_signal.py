from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

from app.source_registry import get_source_info


@dataclass
class NewsSignalResult:
    enabled: bool
    score: float                 # bounded contribution, e.g. -10 to +10
    confidence: float            # 0..1
    regime: str                  # confirmed_news / mixed / noisy / unavailable
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    n_items: int = 0
    n_unique: int = 0
    n_tier1: int = 0
    n_tier2: int = 0
    n_tier3: int = 0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_dt(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)
    s = str(x).strip()
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(s)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _normalize_title_for_fingerprint(title: str) -> str:
    t = (title or "").strip().lower()
    # light normalization (intentionally simple/no regex dependency)
    for ch in ["|", "-", "—", ":", ";", ",", ".", "!", "?", "(", ")", "[", "]", "{", "}", '"', "'"]:
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    return t


def _headline_noise_penalty(title: str) -> float:
    t = (title or "").lower()
    penalty = 0.0
    clickbait_terms = [
        "shocking", "you won't believe", "slams", "destroys", "explodes",
        "rumor", "rumour", "unverified", "viral", "watch:", "must see"
    ]
    for term in clickbait_terms:
        if term in t:
            penalty += 0.8
    if t.count("!") >= 1:
        penalty += 0.5
    return min(3.0, penalty)


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for it in items:
        title = str(it.get("title") or "").strip().lower()
        src = str(it.get("source") or "").strip().lower()
        url = str(it.get("url") or "").strip().lower()

        key = url or f"{src}|{title}"
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _count_near_duplicate_headlines(items: list[dict[str, Any]]) -> int:
    """
    Counts repeated normalized headlines across items (syndication/noise proxy).
    Example:
      same/similar headline reposted across many low-tier sites.
    """
    counts: dict[str, int] = {}
    for it in items:
        fp = _normalize_title_for_fingerprint(str(it.get("title") or ""))
        if not fp:
            continue
        counts[fp] = counts.get(fp, 0) + 1

    # count duplicate *extra* occurrences, not unique keys
    duplicate_excess = 0
    for _, n in counts.items():
        if n > 1:
            duplicate_excess += (n - 1)
    return duplicate_excess


def _recency_weight(published_at: Any, now_utc: Optional[dt.datetime] = None) -> float:
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    pub = _parse_dt(published_at)
    if pub is None:
        return 0.35

    # Future timestamps can happen due to bad feeds / timezone bugs.
    # Treat as low-confidence instead of high-confidence.
    age_minutes = (now_utc - pub).total_seconds() / 60.0
    if age_minutes < -5:
        return 0.25
    age_minutes = max(0.0, age_minutes)

    if age_minutes <= 30:
        return 1.00
    if age_minutes <= 120:
        return 0.85
    if age_minutes <= 360:
        return 0.65
    if age_minutes <= 1440:
        return 0.45
    return 0.25


def _market_move_exhaustion_penalty(market_snapshot: dict[str, Any]) -> float:
    """
    Penalize news chasing if market already appears to have moved hard.
    Expects optional fields if you later add them:
      - recent_move_cents
      - spread_cents
    """
    recent_move = abs(_safe_float(market_snapshot.get("recent_move_cents"), 0.0))
    spread = _safe_float(market_snapshot.get("spread_cents"), 0.0)

    penalty = 0.0
    if recent_move >= 8:
        penalty += 2.0
    elif recent_move >= 5:
        penalty += 1.0

    if spread >= 6:
        penalty += 1.5
    elif spread >= 4:
        penalty += 0.75

    return min(4.0, penalty)


def _extract_fact_weight(item: dict[str, Any]) -> float:
    """
    Placeholder for structured fact extraction.
    For now uses simple metadata flags if present.
    """
    if item.get("official_confirmation") is True:
        return 1.0
    if item.get("secondary_confirmation") is True:
        return 0.6
    return 0.35


def evaluate_news_signal(
    market_snapshot: dict[str, Any],
    news_items: list[dict[str, Any]],
    *,
    enabled: bool = True,
    max_abs_score: float = 10.0,
    require_two_credible_sources_for_boost: bool = True,
    category_allowlist: Optional[list[str]] = None,
) -> NewsSignalResult:
    """
    Bounded news/trend confirmation signal.

    news_items schema (flexible):
      [
        {
          "title": "...",
          "url": "...",
          "source": "Reuters",
          "domain": "reuters.com",            # optional; inferred from url if missing
          "published_at": iso_string,         # or published_ts
          "direction_hint": "YES|NO|NEUTRAL", # optional
          "strength": 0..1,                   # optional
          "official_confirmation": bool,      # optional
          "secondary_confirmation": bool,     # optional
        }
      ]
    """
    if not enabled:
        return NewsSignalResult(
            enabled=False,
            score=0.0,
            confidence=0.0,
            regime="disabled",
            reasons=["news_disabled"],
        )

    category = str(market_snapshot.get("category") or "").strip().lower()
    if category_allowlist:
        allowed = [c.lower() for c in category_allowlist]
        if category and category not in allowed:
            return NewsSignalResult(
                enabled=True,
                score=0.0,
                confidence=0.0,
                regime="category_disabled",
                reasons=[f"news_disabled_for_category:{category}"],
            )

    if not news_items:
        return NewsSignalResult(
            enabled=True,
            score=0.0,
            confidence=0.0,
            regime="unavailable",
            reasons=["no_news_items"],
        )

    items = _dedupe_items(news_items)
    if not items:
        return NewsSignalResult(
            enabled=True,
            score=0.0,
            confidence=0.0,
            regime="unavailable",
            reasons=["all_items_deduped"],
        )

    weighted_yes = 0.0
    weighted_no = 0.0
    weighted_neutral = 0.0
    total_quality = 0.0
    risk_flags: list[str] = []
    reasons: list[str] = []

    tier1_count = tier2_count = tier3_count = 0
    credible_sources = set()
    credible_yes_sources = set()
    credible_no_sources = set()

    now_utc = dt.datetime.now(dt.timezone.utc)

    for it in items:
        src_val = str(it.get("domain") or it.get("url") or it.get("source") or "")
        src = get_source_info(src_val)

        if src.tier == 1:
            tier1_count += 1
        elif src.tier == 2:
            tier2_count += 1
        else:
            tier3_count += 1

        if src.tier in (1, 2):
            credible_sources.add(src.domain)

        recency = _recency_weight(it.get("published_at") or it.get("published_ts"), now_utc=now_utc)
        fact_weight = _extract_fact_weight(it)
        clickbait_penalty = _headline_noise_penalty(str(it.get("title") or ""))

        # strength defaults to moderate
        strength = max(0.0, min(1.0, _safe_float(it.get("strength"), 0.5)))

        # quality bounded per-item
        item_quality = (src.weight * recency * (0.5 + fact_weight)) - (0.15 * clickbait_penalty)
        item_quality = max(0.0, min(1.5, item_quality))
        total_quality += item_quality

        direction = str(it.get("direction_hint") or "NEUTRAL").upper().strip()
        if direction == "YES":
            weighted_yes += item_quality * strength
            if src.tier in (1, 2):
                credible_yes_sources.add(src.domain)
        elif direction == "NO":
            weighted_no += item_quality * strength
            if src.tier in (1, 2):
                credible_no_sources.add(src.domain)
        else:
            weighted_neutral += item_quality * max(0.25, strength)

        # flags
        if src.tier == 3:
            risk_flags.append("tier3_source_present")
        if clickbait_penalty >= 1.5:
            risk_flags.append("clickbait_like_headline_present")
        if recency <= 0.25:
            risk_flags.append("stale_or_bad_timestamp_present")

    # Syndication / repeated-headline penalty (helps avoid fake "consensus")
    duplicate_headline_excess = _count_near_duplicate_headlines(items)
    if duplicate_headline_excess > 0:
        reasons.append(f"headline_dup_excess={duplicate_headline_excess}")
        risk_flags.append("headline_syndication_or_duplication")

        # modest penalty only; don't nuke legitimate wire pickup
        total_quality *= max(0.75, 1.0 - min(0.20, duplicate_headline_excess * 0.03))
        weighted_yes *= max(0.80, 1.0 - min(0.15, duplicate_headline_excess * 0.025))
        weighted_no *= max(0.80, 1.0 - min(0.15, duplicate_headline_excess * 0.025))

    # Agreement / conflict
    directional_total = weighted_yes + weighted_no
    if directional_total <= 0.01:
        reasons.append("no_directional_news_edge")
        raw_score = 0.0
        direction_bias = 0.0
        imbalance = 0.0
    else:
        direction_bias = (weighted_yes - weighted_no) / max(0.001, directional_total)
        raw_score = direction_bias * min(max_abs_score, 2.0 + total_quality * 2.5)

        imbalance = abs(weighted_yes - weighted_no) / max(0.001, directional_total)

        # conflicting reporting penalty if both sides have substantial support
        if directional_total > 0.6 and imbalance < 0.35:
            raw_score *= 0.5
            reasons.append("conflicting_reports_penalty")
            risk_flags.append("conflicting_reports")

        # if lots of neutral relative to directional, confidence/signal should be softer
        if weighted_neutral > directional_total * 0.75:
            raw_score *= 0.85
            reasons.append("high_neutral_share_penalty")
            risk_flags.append("high_neutral_news_share")

    # Require multiple credible confirmations for strong directional boost/cut (symmetrical)
    if require_two_credible_sources_for_boost:
        if raw_score > 0 and len(credible_yes_sources) < 2:
            raw_score = min(raw_score, 2.0)
            reasons.append("limited_credible_yes_confirmation_cap")
            risk_flags.append("insufficient_credible_yes_sources")
        elif raw_score < 0 and len(credible_no_sources) < 2:
            raw_score = max(raw_score, -2.0)
            reasons.append("limited_credible_no_confirmation_cap")
            risk_flags.append("insufficient_credible_no_sources")

    # Market already moved? penalize chasing mostly when following the move.
    # Since we don't know move direction here, apply full penalty only to positive score,
    # and reduced penalty to negative score (keeps shorts/NO signals from being over-muted).
    chase_penalty = _market_move_exhaustion_penalty(market_snapshot)
    if chase_penalty > 0:
        if raw_score > 0:
            raw_score -= chase_penalty
        elif raw_score < 0:
            raw_score -= (0.25 * chase_penalty)  # smaller effect than before
        reasons.append(f"market_move_exhaustion_penalty:{chase_penalty:.1f}")

    # Bound final score
    score = max(-max_abs_score, min(max_abs_score, raw_score))

    # Confidence: combine quality, source diversity, and directional agreement
    quality_norm = max(0.0, min(1.0, total_quality / 4.0))
    source_diversity = max(0.0, min(1.0, len(credible_sources) / 3.0))

    if directional_total <= 0.01:
        agreement_norm = 0.0
    else:
        agreement_norm = max(0.0, min(1.0, abs(direction_bias)))  # 0 mixed, 1 one-sided

    confidence = round(
        max(
            0.0,
            min(
                1.0,
                0.45 * quality_norm + 0.30 * source_diversity + 0.25 * agreement_norm
            ),
        ),
        3,
    )

    # confidence dampeners for noisy conditions
    if "conflicting_reports" in risk_flags:
        confidence = round(max(0.0, confidence * 0.80), 3)
    if "clickbait_like_headline_present" in risk_flags and tier1_count == 0:
        confidence = round(max(0.0, confidence * 0.85), 3)

    # Regime classification
    if abs(score) < 1.5:
        regime = "noisy_or_weak"
    elif "conflicting_reports" in risk_flags:
        regime = "mixed"
    elif score > 0:
        regime = "confirmed_news_yes"
    else:
        regime = "confirmed_news_no"

    if tier1_count >= 2:
        reasons.append("2+_tier1_sources")
    elif tier1_count >= 1:
        reasons.append("1_tier1_source")

    reasons.append(f"credible_sources={len(credible_sources)}")
    reasons.append(f"credible_yes_sources={len(credible_yes_sources)}")
    reasons.append(f"credible_no_sources={len(credible_no_sources)}")
    reasons.append(f"news_items={len(news_items)}")
    reasons.append(f"news_unique={len(items)}")

    # dedupe risk flags
    risk_flags = sorted(set(risk_flags))

    return NewsSignalResult(
        enabled=True,
        score=round(score, 3),
        confidence=confidence,
        regime=regime,
        reasons=reasons,
        risk_flags=risk_flags,
        n_items=len(news_items),
        n_unique=len(items),
        n_tier1=tier1_count,
        n_tier2=tier2_count,
        n_tier3=tier3_count,
    )
