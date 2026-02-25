from __future__ import annotations

from typing import Any, Optional

import httpx


class DiscordNotifier:
    """
    Lightweight Discord webhook notifier.

    Enhancements:
    - Optional richer formatting helpers for trade/skip/exit messages (still plain content).
    - Safe truncation (Discord hard limit is 2000 chars).
    - Basic retry/backoff for transient webhook failures (429/5xx/timeouts).
    - No strategy logic here: only formatting + delivery.
    """

    MAX_LEN = 2000
    SAFE_LEN = 1900  # keep headroom for code fences / extra formatting

    def __init__(self, webhook_url: str, logger=None) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.logger = logger

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def _truncate(self, s: str, limit: int = SAFE_LEN) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 1)] + "…"

    def _post(self, payload: dict) -> None:
        if not self.webhook_url:
            return

        # Minimal retry for transient issues
        attempts = 3
        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                resp = httpx.post(self.webhook_url, json=payload, timeout=10)
                # Success
                if resp.status_code < 400:
                    return

                # Rate limit / transient
                if resp.status_code in (429, 500, 502, 503, 504):
                    if self.logger:
                        self.logger.warning(
                            f"Discord webhook transient error (attempt {i+1}/{attempts}): "
                            f"{resp.status_code} {resp.text[:200]}"
                        )
                    continue

                # Hard failure
                if self.logger:
                    self.logger.warning(f"Discord webhook failed: {resp.status_code} {resp.text}")
                return

            except Exception as e:
                last_err = e
                if self.logger:
                    self.logger.warning(f"Discord send error (attempt {i+1}/{attempts}): {e}")
                continue

        if last_err and self.logger:
            self.logger.warning(f"Discord send failed after retries: {last_err}")

    def send(self, content: str) -> None:
        """
        Backwards-compatible simple sender.
        """
        if not self.webhook_url:
            return
        msg = self._truncate(content, self.SAFE_LEN)
        self._post({"content": msg})

    # -----------------------------
    # Optional helpers you can use from main.py (no breaking changes)
    # -----------------------------
    def send_decision(
        self,
        *,
        kind: str,
        ticker: str,
        market_label: Optional[str] = None,
        direction: Optional[str] = None,
        action: Optional[str] = None,
        stake_dollars: Optional[float] = None,
        count: Optional[int] = None,
        price_cents: Optional[int] = None,
        conviction: Optional[float] = None,
        extra: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        """
        Richer decision message (still plain webhook content).

        kind: "trade" | "exit" | "skip" | "error" | "summary"
        """
        if not self.webhook_url:
            return

        k = (kind or "").lower().strip()
        emoji = {
            "trade": "📈",
            "exit": "📉",
            "skip": "⏭️",
            "error": "⚠️",
            "summary": "📊",
        }.get(k, "ℹ️")

        label = (market_label or "").strip()
        if not label:
            label = ticker

        parts = [f"{emoji} `{ticker}`", f"**{label}**"]

        if action:
            parts.append(f"action={action}")
        if direction:
            parts.append(f"dir={direction}")
        if conviction is not None:
            parts.append(f"conv={float(conviction):.1f}")
        if stake_dollars is not None:
            parts.append(f"stake=${float(stake_dollars):.2f}")
        if count is not None:
            parts.append(f"count={int(count)}")
        if price_cents is not None:
            parts.append(f"px={int(price_cents)}c")
        if dry_run is not None:
            parts.append(f"dry_run={bool(dry_run)}")

        msg = " | ".join(parts)
        if extra:
            msg += "\n" + str(extra).strip()

        self.send(msg)

    def send_codeblock(self, title: str, body: str) -> None:
        """
        Handy for audits / small tables.
        """
        if not self.webhook_url:
            return
        t = (title or "").strip()
        b = (body or "").strip()
        msg = f"**{t}**\n```text\n{b}\n```" if t else f"```text\n{b}\n```"
        self.send(msg)