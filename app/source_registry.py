from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceInfo:
    domain: str
    tier: int               # 1 = highest trust, 3 = lower trust/supportive only
    weight: float           # bounded trust weight
    category_hint: str = "" # optional


# Keep this small + editable. Expand over time.
# Tier 1 examples include official/primary sources.
_SOURCE_MAP: dict[str, SourceInfo] = {
    # Official / primary / high-trust
    "api.weather.gov": SourceInfo("api.weather.gov", tier=1, weight=1.00, category_hint="weather"),
    "weather.gov": SourceInfo("weather.gov", tier=1, weight=1.00, category_hint="weather"),
    "www.fec.gov": SourceInfo("www.fec.gov", tier=1, weight=1.00, category_hint="politics"),
    "fec.gov": SourceInfo("fec.gov", tier=1, weight=1.00, category_hint="politics"),
    "www.sec.gov": SourceInfo("www.sec.gov", tier=1, weight=1.00, category_hint="macro"),
    "sec.gov": SourceInfo("sec.gov", tier=1, weight=1.00, category_hint="macro"),
    "www.cdc.gov": SourceInfo("www.cdc.gov", tier=1, weight=1.00),
    "cdc.gov": SourceInfo("cdc.gov", tier=1, weight=1.00),
    "www.noaa.gov": SourceInfo("www.noaa.gov", tier=1, weight=1.00, category_hint="weather"),
    "noaa.gov": SourceInfo("noaa.gov", tier=1, weight=1.00, category_hint="weather"),
    "www.nhc.noaa.gov": SourceInfo("www.nhc.noaa.gov", tier=1, weight=1.00, category_hint="weather"),
    "nhc.noaa.gov": SourceInfo("nhc.noaa.gov", tier=1, weight=1.00, category_hint="weather"),

    # Major outlets (Tier 2)
    "reuters.com": SourceInfo("reuters.com", tier=2, weight=0.85),
    "www.reuters.com": SourceInfo("www.reuters.com", tier=2, weight=0.85),
    "apnews.com": SourceInfo("apnews.com", tier=2, weight=0.85),
    "www.apnews.com": SourceInfo("www.apnews.com", tier=2, weight=0.85),
    "bloomberg.com": SourceInfo("bloomberg.com", tier=2, weight=0.85),
    "www.bloomberg.com": SourceInfo("www.bloomberg.com", tier=2, weight=0.85),
    "wsj.com": SourceInfo("wsj.com", tier=2, weight=0.80),
    "www.wsj.com": SourceInfo("www.wsj.com", tier=2, weight=0.80),
    "nytimes.com": SourceInfo("nytimes.com", tier=2, weight=0.75),
    "www.nytimes.com": SourceInfo("www.nytimes.com", tier=2, weight=0.75),
    "espn.com": SourceInfo("espn.com", tier=2, weight=0.75, category_hint="sports"),
    "www.espn.com": SourceInfo("www.espn.com", tier=2, weight=0.75, category_hint="sports"),

    # Tier 3 / lower-trust / supportive
    "substack.com": SourceInfo("substack.com", tier=3, weight=0.40),
    "medium.com": SourceInfo("medium.com", tier=3, weight=0.35),
    "x.com": SourceInfo("x.com", tier=3, weight=0.25),
    "twitter.com": SourceInfo("twitter.com", tier=3, weight=0.25),
    "reddit.com": SourceInfo("reddit.com", tier=3, weight=0.20),
    "www.reddit.com": SourceInfo("www.reddit.com", tier=3, weight=0.20),
}


def normalize_domain(url_or_domain: str) -> str:
    s = (url_or_domain or "").strip().lower()
    if not s:
        return ""
    if "://" in s:
        try:
            return urlparse(s).netloc.lower()
        except Exception:
            return s
    return s


def get_source_info(url_or_domain: str) -> SourceInfo:
    dom = normalize_domain(url_or_domain)
    if dom in _SOURCE_MAP:
        return _SOURCE_MAP[dom]

    # fallback heuristic
    if dom.endswith(".gov") or dom.endswith(".mil"):
        return SourceInfo(dom, tier=1, weight=0.95)
    if dom.endswith(".edu"):
        return SourceInfo(dom, tier=2, weight=0.70)
    return SourceInfo(dom or "unknown", tier=3, weight=0.30)
