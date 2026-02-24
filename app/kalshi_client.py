from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.utils import now_ms


class KalshiAPIError(Exception):
    pass


class KalshiAuthError(KalshiAPIError):
    pass


class KalshiRateLimitError(KalshiAPIError):
    pass


@dataclass
class KalshiClient:
    api_key_id: str
    private_key_pem: str
    base_url: str
    dry_run: bool = True
    timeout: float = 15.0
    logger: Any = None
    max_retries: int = 3

    def __post_init__(self) -> None:
        pem = self.private_key_pem.encode("utf-8")
        self._private_key = serialization.load_pem_private_key(pem, password=None)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    # ---------- auth/signing ----------
    def _sign_request(self, timestamp_ms: str, method: str, path: str) -> str:
        # Kalshi requires signing timestamp + HTTP_METHOD + path_without_query
        path_without_query = path.split("?")[0]
        msg = f"{timestamp_ms}{method.upper()}{path_without_query}".encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _build_headers(self, method: str, path: str, extra: Optional[dict] = None) -> Dict[str, str]:
        ts = now_ms()
        sig = self._sign_request(ts, method, path)
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    # ---------- transport ----------
    def _request(self, method: str, path: str, *, params: Optional[dict] = None, json_body: Optional[dict] = None) -> Any:
        headers = self._build_headers(method, path)
        backoff = 1.0
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.request(method, path, headers=headers, params=params, json=json_body)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else backoff
                    if self.logger:
                        self.logger.warning(f"429 rate limit on {method} {path}; retrying in {wait_s:.1f}s")
                    time.sleep(wait_s)
                    backoff *= 2
                    continue

                if resp.status_code in (401, 403):
                    raise KalshiAuthError(f"Auth failed: {resp.status_code} {resp.text}")

                if resp.status_code >= 400:
                    raise KalshiAPIError(f"HTTP {resp.status_code}: {resp.text}")

                if not resp.text:
                    return {}
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt == self.max_retries:
                    break
                if self.logger:
                    self.logger.warning(f"Network error attempt {attempt}/{self.max_retries}: {e}")
                time.sleep(backoff)
                backoff *= 2

        if last_err:
            raise KalshiAPIError(f"Network error after retries: {last_err}")
        raise KalshiAPIError("Request failed after retries")

    # ---------- common endpoints ----------
    # NOTE: Endpoint shapes may vary as Kalshi evolves; adjust if docs changed.
    def get_portfolio_balance(self) -> Any:
        return self._request("GET", "/trade-api/v2/portfolio/balance")

    def get_positions(self) -> Any:
        return self._request("GET", "/trade-api/v2/portfolio/positions")

    def list_orders(self, limit: int = 50) -> Any:
        return self._request("GET", "/trade-api/v2/portfolio/orders", params={"limit": limit})

    def get_order(self, order_id: str) -> Any:
        return self._request("GET", f"/trade-api/v2/portfolio/orders/{order_id}")

    def list_fills(self, limit: int = 50) -> Any:
        return self._request("GET", "/trade-api/v2/portfolio/fills", params={"limit": limit})

    def get_markets(self, limit: int = 100, status: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", "/trade-api/v2/markets", params=params)

    def get_market(self, ticker: str) -> Any:
        return self._request("GET", f"/trade-api/v2/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> Any:
        # Some deployments/docs may use "orderbook" naming variations.
        return self._request("GET", f"/trade-api/v2/markets/{ticker}/orderbook", params={"depth": depth})

    def list_trades(self, ticker: Optional[str] = None, limit: int = 50) -> Any:
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/trade-api/v2/markets/trades", params=params)

    def place_order(
        self,
        *,
        ticker: str,
        side: str,
        yes_no: str,
        count: int,
        price_cents: int,
        client_order_id: str,
        order_type: str = "limit",
    ) -> Any:
        """
        side / yes_no names may need adjustment to exact schema fields from your current docs.
        This method is conservative and explicit, and DRY_RUN-safe.
        """
        if order_type.lower() != "limit":
            raise ValueError("Only limit orders are supported in this MVP")
        if count <= 0:
            raise ValueError("count must be > 0")
        if not (1 <= price_cents <= 99):
            raise ValueError("price_cents must be between 1 and 99")
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if yes_no.upper() not in {"YES", "NO"}:
            raise ValueError("yes_no must be YES or NO")

        body = {
            "ticker": ticker,
            "side": side.lower(),
            "yes_no": yes_no.upper(),
            "count": count,
            "price": price_cents,
            "type": "limit",
            "client_order_id": client_order_id,
        }

        if self.dry_run:
            return {
                "dry_run": True,
                "simulated": True,
                "path": "/trade-api/v2/portfolio/orders",
                "request": body,
                "message": "DRY_RUN enabled; no live order sent.",
            }

        return self._request("POST", "/trade-api/v2/portfolio/orders", json_body=body)

    def cancel_order(self, order_id: str) -> Any:
        if self.dry_run:
            return {
                "dry_run": True,
                "simulated": True,
                "path": f"/trade-api/v2/portfolio/orders/{order_id}",
                "message": "DRY_RUN enabled; no live cancel sent.",
            }
        return self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
