from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    # exact names user showed in Render
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    DRY_RUN: bool = _bool("DRY_RUN", True)
    KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
    KALSHI_ENV: str = os.getenv("KALSHI_ENV", "demo").strip().lower()
    KALSHI_PRIVATE_KEY_PEM: str = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # tuning
    SCAN_INTERVAL_SECONDS: int = _int("SCAN_INTERVAL_SECONDS", 60)
    MIN_TRADE_DOLLARS: int = _int("MIN_TRADE_DOLLARS", 10)
    MAX_TRADE_DOLLARS: int = _int("MAX_TRADE_DOLLARS", 100)
    MIN_CONVICTION_SCORE: int = _int("MIN_CONVICTION_SCORE", 58)  # slightly risk-on
    MAX_SPREAD_CENTS: int = _int("MAX_SPREAD_CENTS", 8)
    MIN_RECENT_VOLUME: int = _int("MIN_RECENT_VOLUME", 0)

    SOCIAL_SIGNAL_ENABLED: bool = _bool("SOCIAL_SIGNAL_ENABLED", True)
    SOCIAL_MAX_BONUS_SCORE: int = _int("SOCIAL_MAX_BONUS_SCORE", 12)
    SOCIAL_SIZE_BOOST_MAX_PCT: float = _float("SOCIAL_SIZE_BOOST_MAX_PCT", 0.20)

    DAILY_MAX_LOSS_DOLLARS: float = _float("DAILY_MAX_LOSS_DOLLARS", 75.0)
    MAX_OPEN_EXPOSURE_DOLLARS: float = _float("MAX_OPEN_EXPOSURE_DOLLARS", 250.0)
    MAX_SIMULTANEOUS_POSITIONS: int = _int("MAX_SIMULTANEOUS_POSITIONS", 4)

    WITHDRAWAL_ALERT_THRESHOLD_DOLLARS: float = _float("WITHDRAWAL_ALERT_THRESHOLD_DOLLARS", 500.0)

    DB_PATH: str = os.getenv("DB_PATH", "bot_state.db")

    @property
    def base_url(self) -> str:
        # Kalshi quickstart examples use demo-api.kalshi.co and api.kalshi.com
        return "https://demo-api.kalshi.co" if self.KALSHI_ENV == "demo" else "https://api.kalshi.com"

    def validate(self) -> None:
        if not self.KALSHI_API_KEY_ID:
            raise ValueError("Missing KALSHI_API_KEY_ID")
        if not self.KALSHI_PRIVATE_KEY_PEM:
            raise ValueError("Missing KALSHI_PRIVATE_KEY_PEM")
        if self.KALSHI_ENV not in {"demo", "prod", "production"}:
            raise ValueError("KALSHI_ENV must be demo or prod")
