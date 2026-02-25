from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _xml_text(parent: ET.Element, tag: str) -> str:
    el = parent.find(tag)
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _title_fingerprint(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = " ".join(t.split())
    return t


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        fp = _title_fingerprint(str(it.get("title") or ""))
        key = (it.get("url") or "") or fp
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def build_news_query_from_market_label(market_label: str, *, max_terms: int = 12) -> str:
    """
    Convert market label/title/question into a Google News query.
    Keep it simple & robust.
    """
    s = (market_label or "").strip()
    if not s:
        return ""

    # remove punctuation, keep words/numbers
    cleaned = re.sub(r"[^a-zA-Z0-9\s']", " ", s)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    stop = {
        "will", "the", "a", "an", "to", "in", "on", "by", "before", "after", "of", "and", "or",
        "be", "is", "are", "was", "were", "happen", "occur", "occurs", "happens",
        "yes", "no",
    }
    terms = []
    for w in cleaned.split():
        lw = w.lower()
        if lw in stop:
            continue
        terms.append(w)

    if not terms:
        return cleaned[:140]

    return " ".join(terms[:max_terms])[:140]


def fetch_google_news_rss(query: str, *, max_items: int = 15, timeout: float = 6.0) -> List[Dict[str, Any]]:
    """
    Fetch Google News RSS (no API key).
    Returns items with keys used by app.news_signal.evaluate_news_signal():
      - title, url, source, domain, published_at, snippet
    """
    q = (query or "").strip()
    if not q:
        return []

    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = httpx.get(url, timeout=timeout, headers={"User-Agent": "kalshi-bot/1.0"})
        if resp.status_code >= 400:
            return []
        xml_text = resp.text or ""
    except Exception:
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    items: List[Dict[str, Any]] = []
    for it in channel.findall("item")[: max(1, int(max_items))]:
        title = _xml_text(it, "title")
        link = _xml_text(it, "link")
        pub = _xml_text(it, "pubDate")
        desc = _xml_text(it, "description")
        snippet = _strip_html(desc)[:280]

        source_el = it.find("source")
        source = source_el.text.strip() if (source_el is not None and source_el.text) else ""
        domain = ""
        if source_el is not None and source_el.attrib.get("url"):
            domain = source_el.attrib.get("url", "")
        # domain can also be inferred from link later by source_registry

        items.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "domain": domain,
                "published_at": pub,   # may not be ISO; your parser safely handles unknown
                "snippet": snippet,
            }
        )

    return _dedupe(items)


# -----------------------------
# Direction inference -> direction_hint / strength
# -----------------------------

_POS = {"confirmed", "confirm", "evidence", "approved", "wins", "win", "passed", "passes", "announced", "launch", "launched"}
_NEG = {"denied", "deny", "debunked", "debunk", "false", "fake", "hoax", "rejected", "rejects", "blocked", "cancelled", "canceled", "fails", "failed"}
_MIX = {"reportedly", "rumor", "rumour", "may", "might", "could", "unconfirmed", "alleged", "unclear"}


def infer_direction_hint(item: Dict[str, Any], *, yes_means_event_occurs: bool) -> Dict[str, Any]:
    """
    Adds:
      direction_hint: YES/NO/NEUTRAL
      strength: 0..1
    The sign means: YES supports the market's YES outcome (taking question negation into account).
    """
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    text = f"{title} {snippet}".strip()

    pos = sum(1 for w in _POS if w in text)
    neg = sum(1 for w in _NEG if w in text)
    mix = sum(1 for w in _MIX if w in text)

    raw = pos - neg
    if raw == 0:
        item["direction_hint"] = "NEUTRAL"
        item["strength"] = 0.35
        return item

    # Base strength from magnitude
    strength = 0.55 + min(0.25, 0.10 * abs(raw))
    if mix > 0:
        strength -= 0.18

    strength = max(0.10, min(1.0, strength))

    # raw>0 means "event more likely occurs"; raw<0 means "event less likely occurs"
    # If YES means event occurs -> raw>0 => YES
    # If YES means event does NOT occur (negated question) -> flip
    if yes_means_event_occurs:
        direction = "YES" if raw > 0 else "NO"
    else:
        direction = "NO" if raw > 0 else "YES"

    item["direction_hint"] = direction
    item["strength"] = strength
    return item


def yes_means_event_occurs(market_label: str) -> bool:
    """
    If the question is negated ("Will X NOT happen"), then YES means "no event".
    """
    s = (market_label or "").strip().lower()
    if not s:
        return True
    if "will not" in s or "won't" in s:
        return False

    # loose "will ... not ..." window
    toks = re.findall(r"[a-z']+", s)
    for i, t in enumerate(toks):
        if t == "will":
            window = toks[i : i + 6]
            if "not" in window or "never" in window:
                return False
            break
    return True