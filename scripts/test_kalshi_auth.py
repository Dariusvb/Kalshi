from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from app.config import Settings
from app.kalshi_client import KalshiAuthError, KalshiAPIError, KalshiClient
from app.logger import get_logger


def main() -> int:
    load_dotenv()
    settings = Settings()
    logger = get_logger("kalshi-auth-test", settings.LOG_LEVEL)

    try:
        settings.validate()
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return 1

    logger.info(f"Starting auth test | env={settings.KALSHI_ENV} | dry_run={settings.DRY_RUN}")

    client = KalshiClient(
        api_key_id=settings.KALSHI_API_KEY_ID,
        private_key_pem=settings.KALSHI_PRIVATE_KEY_PEM,
        base_url=settings.base_url,
        dry_run=settings.DRY_RUN,
        logger=logger,
    )

    try:
        # Try balance first (authenticated)
        bal = client.get_portfolio_balance()
        keys = list(bal.keys())[:10] if isinstance(bal, dict) else []
        logger.info(f"✅ Auth success. Balance endpoint responded. top_keys={keys}")

        # Try orders list too
        orders = client.list_orders(limit=5)
        okeys = list(orders.keys())[:10] if isinstance(orders, dict) else []
        logger.info(f"✅ Orders endpoint responded. top_keys={okeys}")
        return 0

    except KalshiAuthError as e:
        logger.error(f"❌ Auth failed: {e}")
        logger.error("Check KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PEM formatting, env=demo/prod, and signing code.")
        return 2
    except KalshiAPIError as e:
        logger.error(f"❌ API error: {e}")
        return 3
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        return 4
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
