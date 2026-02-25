from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        self.base_url = self.base_url.rstrip("/")

        pem_text = (self.private_key_pem or "").strip()

        # Support Render/local envs that store PEM with escaped newlines
        # e.g. "-----BEGIN...-----\nLINE\n-----END...-----"
        if "\\n" in pem_text and "\n" not in pem_text:
            pem_text = pem_text.replace("\\n", "\n")

        # Helpful normalization if someone accidentally wrapped in quotes in env
        if (pem_text.startswith('"') and pem_text.endswith('"')) or (
            pem_text.startswith("'") and pem_text.endswith("'")
        ):
            pem_text = pem_text[1:-1]

        try:
            pem_bytes = pem_text.encode("utf-8")
            self._private_key = serialization.load_pem_private_key(pem_bytes, password=None)
        except Exception as e:
            raise KalshiAuthError(
                "Failed to load KALSHI_PRIVATE_KEY_PEM. Ensure full PEM is present "
                "(including BEGIN/END lines) and formatting/newlines are correct."
            ) from e

        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

        if self.logger:
            # Safe debug info only (no secrets)
            self.logger.info(f"KalshiClient initialized | base_url={self.base_url} | dry_run={self.dry_run}")

    def close(self) -> None:
        self._client.close()

    # ---------- auth/signing ----------
    def _sign_request(self, timestamp_ms: str, method: str, path: str) -> str:
        """
        Kalshi signing message = timestamp + HTTP_METHOD + path_without_query
        Signature: RSA-PSS SHA256, base64-encoded
        """
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
            "KALSHI-ACCESS-KEY": self.api_key_id.strip(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    # ---------- transport ----------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        use_auth: bool = True,
    ) -> Any:
        """
        Performs request with retries. Rebuilds auth headers each retry so timestamp/signature stay fresh.
        """
        backoff = 1.0
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                headers = self._build_headers(method, path) if use_auth else {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }

                resp = self._client.request(method, path, headers=headers, params=params, json=json_body)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else backoff
                    if self.logger:
                        self.logger.warning(
                            f"429 rate limit on {method.upper()} {path}; retrying in {wait_s:.1f}s "
                            f"(attempt {attempt}/{self.max_retries})"
                        )
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

            except httpx.RequestError as e:
                # Covers DNS issues like [Errno -2] Name or service not known
                last_err = e
                if attempt == self.max_retries:
                    break
                if self.logger:
                    self.logger.warning(f"Request error attempt {attempt}/{self.max_retries}: {e}")
                time.sleep(backoff)
                backoff *= 2

        if last_err:
            raise KalshiAPIError(f"Network error after retries: {last_err}")
        raise KalshiAPIError("Request failed after retries")

    # ---------- public helpers ----------
    def public_markets_ping(self) -> Any:
        """Unauthenticated smoke test helper."""
        return self._request("GET", "/trade-api/v2/markets", params={"limit": 1}, use_auth=False)

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

    def get_markets(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Any:
        params: Dict[str, Any] = {"limit": int(limit)}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        # Try authenticated request first. If auth blocks or key is mispaired,
        # public fallback is helpful for smoke tests / debugging.
        try:
            return self._request("GET", "/trade-api/v2/markets", params=params, use_auth=True)
        except KalshiAuthError:
            if self.logger:
                self.logger.warning("Auth failed on markets endpoint; trying public fallback.")
            return self._request("GET", "/trade-api/v2/markets", params=params, use_auth=False)

    def get_market(self, ticker: str) -> Any:
        if not ticker:
            raise ValueError("ticker is required")
        return self._request("GET", f"/trade-api/v2/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> Any:
        if not ticker:
            raise ValueError("ticker is required")
        depth = max(1, int(depth))
        # Some deployments/docs may use "orderbook" naming variations.
        return self._request("GET", f"/trade-api/v2/markets/{ticker}/orderbook", params={"depth": depth})

    def list_trades(self, ticker: Optional[str] = None, limit: int = 50) -> Any:
        params: Dict[str, Any] = {"limit": int(limit)}
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
        side / yes_no names may need adjustment to exact schema fields from current docs.
        DRY_RUN-safe. Validates inputs before simulating.
        """
        if not ticker:
            raise ValueError("ticker is required")
        if not client_order_id:
            raise ValueError("client_order_id is required")
        if order_type.lower() != "limit":
            raise ValueError("Only limit orders are supported in this MVP")
        if int(count) <= 0:
            raise ValueError("count must be > 0")
        if not (1 <= int(price_cents) <= 99):
            raise ValueError("price_cents must be between 1 and 99")
        if side.lower() not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        if yes_no.upper() not in {"YES", "NO"}:
            raise ValueError("yes_no must be YES or NO")

        body = {
            "ticker": ticker,
            "side": side.lower(),
            "yes_no": yes_no.upper(),
            "count": int(count),
            "price": int(price_cents),
            "type": "limit",
            "client_order_id": client_order_id,
        }

        if self.dry_run:
            if self.logger:
                self.logger.info(f"DRY_RUN place_order simulated for {ticker} {yes_no.upper()} {side.lower()}")
            return {
                "dry_run": True,
                "simulated": True,
                "path": "/trade-api/v2/portfolio/orders",
                "request": body,
                "message": "DRY_RUN enabled; no live order sent.",
            }

        return self._request("POST", "/trade-api/v2/portfolio/orders", json_body=body)

    def cancel_order(self, order_id: str) -> Any:
        if not order_id:
            raise ValueError("order_id is required")

        if self.dry_run:
            if self.logger:
                self.logger.info(f"DRY_RUN cancel_order simulated for {order_id}")
            return {
                "dry_run": True,
                "simulated": True,
                "path": f"/trade-api/v2/portfolio/orders/{order_id}",
                "message": "DRY_RUN enabled; no live cancel sent.",
            }

        return self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
