from __future__ import annotations

from typing import Optional

import httpx


class DiscordNotifier:
    def __init__(self, webhook_url: str, logger=None) -> None:
        self.webhook_url = webhook_url.strip()
        self.logger = logger

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, content: str) -> None:
        if not self.webhook_url:
            return
        try:
            resp = httpx.post(self.webhook_url, json={"content": content[:1900]}, timeout=10)
            if resp.status_code >= 400 and self.logger:
                self.logger.warning(f"Discord webhook failed: {resp.status_code} {resp.text}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Discord send error: {e}")
