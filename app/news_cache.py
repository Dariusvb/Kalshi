from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NewsCache:
    path: str
    ttl_seconds: int = 6 * 3600
    _data: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load()
        self._prune()

    def _load(self) -> None:
        if not self.path:
            return
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._data = {str(k): float(v) for k, v in raw.items()}
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            pass

    def _prune(self) -> None:
        now = time.time()
        keep = {}
        for k, ts in self._data.items():
            if now - ts <= self.ttl_seconds:
                keep[k] = ts
        self._data = keep
        self._save()

    @staticmethod
    def make_key(item: dict[str, Any]) -> str:
        # Try URL first, then title+source+published
        url = str(item.get("url") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        source = str(item.get("source") or "").strip().lower()
        published = str(item.get("published_ts") or item.get("published_at") or "").strip().lower()
        raw = f"{url}|{title}|{source}|{published}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def seen_recently(self, item: dict[str, Any]) -> bool:
        self._prune()
        k = self.make_key(item)
        ts = self._data.get(k)
        if ts is None:
            return False
        return (time.time() - ts) <= self.ttl_seconds

    def mark_seen(self, item: dict[str, Any]) -> None:
        self._prune()
        k = self.make_key(item)
        self._data[k] = time.time()
        self._save()
