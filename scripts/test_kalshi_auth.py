from __future__ import annotations

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
    logger.info(f"Resolved base_url={settings.base_url}")

    client = KalshiClient(
        api_key_id=settings.KALSHI_API_KEY_ID,
        private_key_pem=settings.KALSHI_PRIVATE_KEY_PEM,
        base_url=settings.base_url,
        dry_run=settings.DRY_RUN,
        logger=logger,
    )

    try:
        # --- Public endpoint smoke test first (helps isolate DNS/base-url issues) ---
        public = client.get_markets(limit=1)
        if isinstance(public, dict):
            pkeys = list(public.keys())[:10]
            logger.info(f"✅ Public markets endpoint responded. top_keys={pkeys}")
        else:
            logger.info("✅ Public markets endpoint responded.")

        # --- Authenticated endpoints ---
        bal = client.get_portfolio_balance()
        bkeys = list(bal.keys())[:10] if isinstance(bal, dict) else []
        logger.info(f"✅ Auth success. Balance endpoint responded. top_keys={bkeys}")

        orders = client.list_orders(limit=5)
        okeys = list(orders.keys())[:10] if isinstance(orders, dict) else []
        logger.info(f"✅ Orders endpoint responded. top_keys={okeys}")
        return 0

    except KalshiAuthError as e:
        logger.error(f"❌ Auth failed: {e}")
        logger.error("Check KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PEM formatting, key/env pairing (demo vs prod), and signing code.")
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
