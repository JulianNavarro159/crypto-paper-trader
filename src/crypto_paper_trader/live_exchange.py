from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Sequence
from urllib.parse import urlencode

import httpx
from websockets.sync.client import connect as ws_sync_connect

from crypto_paper_trader.config import ExchangeConfig

logger = logging.getLogger(__name__)


class LiveExchangeExecutionError(RuntimeError):
    """Exception raised when an authenticated Binance Spot API order fails."""


def quantize_step_size(quantity: float, step_size: str) -> str:
    """Round a base asset quantity down to the exact Binance LOT_SIZE step.

    Args:
        quantity: Raw floating-point asset quantity.
        step_size: Decimal step string returned by Binance exchangeInfo.

    Returns:
        Formatted quantity string compliant with Binance LOT_SIZE filter.
    """
    qty_dec = Decimal(str(quantity))
    step_dec = Decimal(step_size)
    if step_dec <= Decimal("0"):
        return f"{quantity:.8f}"
    quantized = (qty_dec // step_dec) * step_dec
    return format(quantized.quantize(step_dec, rounding=ROUND_DOWN), "f")


class BinanceSpotAuthenticatedClient:
    """Authenticated Binance Spot REST client for real-money autonomous trading.

    Args:
        config: Exchange connectivity configuration.
        api_key: Binance Spot API key with Spot Trading permission enabled.
        api_secret: Binance Spot API secret key for HMAC-SHA256 signing.
        timeout_seconds: HTTP request timeout in seconds.
    """

    _ip_cooldown_until: float = 0.0

    @classmethod
    def is_cooldown_active(cls) -> bool:
        """Return True if Binance authenticated account REST is in 418 cooldown."""
        return time.time() < cls._ip_cooldown_until

    def __init__(
        self,
        config: ExchangeConfig,
        api_key: str,
        api_secret: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._config = config
        self._api_key = api_key.strip()
        self._api_secret = api_secret.strip()
        self._timeout_seconds = timeout_seconds
        self._step_size_cache: dict[str, str] = {}
        self._balance_cache: dict[str, tuple[float, float]] = {}
        self._last_known_usdt_balance: float | None = None

    def get_known_usdt_balance(self) -> float | None:
        """Return the most recently observed or updated live USDT balance."""
        return self._last_known_usdt_balance

    def update_known_usdt_balance(self, usdt_balance: float) -> None:
        """Update the tracked live USDT balance after an order fill."""
        self._last_known_usdt_balance = max(0.0, round(usdt_balance, 6))

    def get_known_asset_balance(self, asset: str) -> float:
        """Return the last known free balance for an asset without a REST call."""
        entry = self._balance_cache.get(asset.upper())
        return entry[1] if entry is not None else 0.0

    def adjust_known_asset_balance(self, asset: str, delta: float) -> None:
        """Adjust the tracked free balance for an asset after an order fill."""
        upper = asset.upper()
        current = self.get_known_asset_balance(upper)
        self._balance_cache[upper] = (
            time.time(),
            max(0.0, round(current + delta, 8)),
        )

    @classmethod
    def from_env(cls, config: ExchangeConfig) -> BinanceSpotAuthenticatedClient:
        """Instantiate an authenticated Binance client using environment variables.

        Args:
            config: Exchange connectivity configuration.

        Returns:
            Configured BinanceSpotAuthenticatedClient instance.

        Raises:
            LiveExchangeExecutionError: If BINANCE_API_KEY or SECRET is missing.
        """
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            raise LiveExchangeExecutionError(
                "Missing BINANCE_API_KEY or BINANCE_API_SECRET in environment."
            )
        return cls(config=config, api_key=api_key, api_secret=api_secret)

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Append timestamp and HMAC-SHA256 signature to request parameters.

        Args:
            params: Unsigned parameter dictionary.

        Returns:
            Signed parameter dictionary ready for Binance REST transmission.
        """
        signed = dict(params)
        signed["timestamp"] = int(time.time() * 1000)
        signed["recvWindow"] = 5000
        query_string = urlencode(signed)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed["signature"] = signature
        return signed

    def _headers(self) -> dict[str, str]:
        """Return HTTP headers containing the Binance API key."""
        return {"X-MBX-APIKEY": self._api_key}

    def _sign_ws_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sign request parameters alphabetically for Binance WebSocket API v3.

        Args:
            params: Unsigned parameter dictionary.

        Returns:
            Signed parameter dictionary containing apiKey, timestamp, and signature.
        """
        ws_params: dict[str, Any] = dict(params)
        ws_params["apiKey"] = self._api_key
        ws_params["recvWindow"] = 5000
        ws_params["timestamp"] = int(time.time() * 1000)
        sorted_items = sorted(
            (
                key,
                "true" if val is True else ("false" if val is False else val),
            )
            for key, val in ws_params.items()
        )
        query_string = urlencode(sorted_items)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        ws_params["signature"] = signature
        return ws_params

    def _request_via_ws_api(
        self, method: str, path: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute an authenticated request over Binance WebSocket API v3.

        Args:
            method: HTTP method (GET or POST).
            path: REST endpoint path mapped to a WebSocket API v3 method.
            params: Unsigned request parameters.

        Returns:
            Decoded response dictionary if successful, otherwise None.
        """
        ws_method_map: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
            ("GET", "/api/v3/account"): (
                "account.status",
                {"omitZeroBalances": True},
            ),
            ("GET", "/api/v3/myTrades"): ("myTrades", {}),
            ("POST", "/api/v3/order"): ("order.place", {}),
        }
        key = (method.upper(), path)
        if key not in ws_method_map:
            return None

        ws_method, extra_params = ws_method_map[key]
        merged_params = {**params, **extra_params}
        signed_ws_params = self._sign_ws_params(merged_params)
        ws_endpoints = (
            "wss://ws-api.binance.com:443/ws-api/v3",
            "wss://ws-api.binance.com:9443/ws-api/v3",
        )
        req_payload = json.dumps(
            {
                "id": f"auth-{int(time.time() * 1000)}",
                "method": ws_method,
                "params": signed_ws_params,
            }
        )
        for ws_url in ws_endpoints:
            try:
                with ws_sync_connect(
                    ws_url,
                    open_timeout=self._timeout_seconds,
                    close_timeout=5.0,
                ) as ws_conn:
                    ws_conn.send(req_payload)
                    raw_msg = ws_conn.recv(timeout=self._timeout_seconds)
                    decoded = json.loads(raw_msg)
                    if decoded.get("status") == 200 and "result" in decoded:
                        res_data = decoded["result"]
                        if isinstance(res_data, dict):
                            return res_data
                        if isinstance(res_data, list):
                            return {"result": res_data}
            except Exception as exc:
                logger.debug(
                    "WebSocket API v3 %s failed on %s: %s",
                    ws_method,
                    ws_url,
                    exc,
                )
        return None

    def _request_with_failover(
        self, method: str, path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute an authenticated request with WS API v3 and cluster failover.

        Args:
            method: HTTP method (GET or POST).
            path: Relative API endpoint path such as /api/v3/account.
            params: Unsigned parameter dictionary.

        Returns:
            Decoded JSON dictionary response from the successful cluster.

        Raises:
            LiveExchangeExecutionError: If all cluster endpoints fail.
        """
        now_ts = time.time()
        is_order_post = method.upper() == "POST" and path == "/api/v3/order"
        if (
            not is_order_post
            and now_ts < BinanceSpotAuthenticatedClient._ip_cooldown_until
        ):
            wait_rem = int(BinanceSpotAuthenticatedClient._ip_cooldown_until - now_ts)
            raise LiveExchangeExecutionError(
                f"Binance IP 418 cooldown active; skipping request for {wait_rem}s"
            )

        urls = [
            "https://api-gcp.binance.com",
            "https://api4.binance.com",
            "https://api3.binance.com",
            "https://api2.binance.com",
            "https://api1.binance.com",
            self._config.rest_base_url,
        ]
        last_error: Exception | None = None
        for idx, base_url in enumerate(urls):
            endpoint = f"{base_url.rstrip('/')}{path}"
            signed = self._sign_params(params)
            headers = dict(self._headers())
            headers["User-Agent"] = "CryptoPaperTrader/1.0"
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    if method.upper() == "GET":
                        response = client.get(endpoint, params=signed, headers=headers)
                    else:
                        response = client.post(endpoint, params=signed, headers=headers)
                    if response.status_code == 418:
                        if is_order_post and idx == 0:
                            logger.warning(
                                "Binance 418 on %s during order POST; "
                                "trying backup cluster %s once...",
                                base_url,
                                urls[1],
                            )
                            continue
                        ws_fallback = self._request_via_ws_api(method, path, params)
                        if ws_fallback is not None:
                            return ws_fallback
                        retry_hdr = response.headers.get("Retry-After", "240").strip()
                        cooldown_secs = (
                            max(float(retry_hdr) + 15.0, 240.0)
                            if retry_hdr.isdigit()
                            else 240.0
                        )
                        BinanceSpotAuthenticatedClient._ip_cooldown_until = (
                            time.time() + cooldown_secs
                        )
                        logger.warning(
                            "Binance IP ban 418 on %s (Retry-After=%s); "
                            "initiating %.0fs quiet cooldown...",
                            base_url,
                            retry_hdr,
                            cooldown_secs,
                        )
                        raise LiveExchangeExecutionError(
                            f"Binance IP ban 418 encountered; "
                            f"cooldown for {int(cooldown_secs)}s"
                        )
                    if response.status_code == 429:
                        last_error = httpx.HTTPStatusError(
                            f"Rate limited: HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                        logger.warning(
                            "Binance rate limit 429 on %s; "
                            "trying fallback cluster...",
                            base_url,
                        )
                        time.sleep(2.0)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict):
                        return data
                    if isinstance(data, list):
                        return {"result": data}
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Authenticated request failed on %s (%s); retrying...",
                    base_url,
                    exc,
                )
                time.sleep(0.5)

        raise LiveExchangeExecutionError(
            f"All Binance authenticated clusters failed for {path}: {last_error}"
        )

    def fetch_available_asset_balance(self, asset: str) -> float:
        """Query the user real Binance Spot account for free balance of an asset.

        Args:
            asset: Asset ticker symbol in uppercase such as 'USDT' or 'SUI'.

        Returns:
            Available unencumbered balance of the requested asset in Spot wallet.

        Raises:
            LiveExchangeExecutionError: If the authenticated account query fails.
        """
        upper_asset = asset.upper()
        now = time.time()
        if upper_asset in self._balance_cache:
            cached_time, cached_val = self._balance_cache[upper_asset]
            if now - cached_time < 180.0:
                return cached_val

        payload = self._request_with_failover("GET", "/api/v3/account", {})
        for asset_entry in payload.get("balances", []):
            cur_asset = str(asset_entry.get("asset", "")).upper()
            cur_free = float(asset_entry.get("free", 0.0))
            self._balance_cache[cur_asset] = (now, cur_free)
            if cur_asset == "USDT":
                self._last_known_usdt_balance = round(cur_free, 6)
            if cur_asset == upper_asset:
                return cur_free

        self._balance_cache[upper_asset] = (now, 0.0)
        return 0.0

    def fetch_available_usdt_balance(self) -> float:
        """Query the user real Binance Spot account for free USDT balance.

        Returns:
            Available unencumbered USDT balance in Spot wallet.

        Raises:
            LiveExchangeExecutionError: If the authenticated account query fails.
        """
        return self.fetch_available_asset_balance("USDT")

    def fetch_active_spot_holding(
        self,
        universe_symbols: Sequence[str],
    ) -> tuple[str, float, float] | None:
        """Detect an existing non-USDT holding in the user Spot account.

        Args:
            universe_symbols: Trading pair symbols to inspect such as SUIUSDT.

        Returns:
            Tuple of (symbol, quantity, entry_price) if a holding >= 5 USDT exists,
            otherwise None.
        """
        try:
            payload = self._request_with_failover("GET", "/api/v3/account", {})
        except LiveExchangeExecutionError:
            return None

        now = time.time()
        balances_by_asset = {
            str(entry.get("asset", "")).upper(): float(entry.get("free", 0.0))
            for entry in payload.get("balances", [])
        }
        for asset_code, free_val in balances_by_asset.items():
            self._balance_cache[asset_code] = (now, free_val)
        if "USDT" in balances_by_asset:
            self._last_known_usdt_balance = round(balances_by_asset["USDT"], 6)

        for sym in universe_symbols:
            upper_sym = sym.upper()
            if not upper_sym.endswith("USDT"):
                continue
            base_asset = upper_sym[:-4]
            qty = balances_by_asset.get(base_asset, 0.0)
            if qty <= 0.0:
                continue

            try:
                trades_resp = self._request_with_failover(
                    "GET", "/api/v3/myTrades", {"symbol": upper_sym, "limit": 5}
                )
                trades = trades_resp.get("result", trades_resp)
            except LiveExchangeExecutionError:
                continue

            if isinstance(trades, list) and trades:
                last_trade = trades[-1]
                if not last_trade.get("isBuyer"):
                    continue
                entry_price = float(last_trade.get("price", 0.0))
                last_order_id = last_trade.get("orderId")
                last_buy_qty = sum(
                    float(t.get("qty", 0.0))
                    for t in trades
                    if isinstance(t, dict)
                    and t.get("isBuyer")
                    and t.get("orderId") == last_order_id
                )
                if last_buy_qty <= 0.0:
                    last_buy_qty = float(last_trade.get("qty", qty))
                step_size = self.fetch_symbol_step_size(upper_sym)
                raw_tradeable = (
                    max(0.0, qty - 0.0008)
                    if base_asset == "BNB"
                    else min(qty, last_buy_qty)
                )
                tradeable_qty = float(quantize_step_size(raw_tradeable, step_size))
                if entry_price > 0.0 and (tradeable_qty * entry_price) >= 12.0:
                    return (upper_sym, tradeable_qty, entry_price)
        return None

    def fetch_symbol_step_size(self, symbol: str) -> str:
        """Fetch LOT_SIZE stepSize for a symbol from Binance exchangeInfo.

        Args:
            symbol: Trading pair symbol such as SUIUSDT.

        Returns:
            Step size string such as '0.10000000' or '0.00001000'.
        """
        upper_sym = symbol.upper()
        if upper_sym in self._step_size_cache:
            return self._step_size_cache[upper_sym]

        urls = [
            "https://api.binance.us",
            "https://api-gcp.binance.com",
            "https://api4.binance.com",
            "https://data-api.binance.vision",
            "https://api1.binance.com",
            "https://api2.binance.com",
            self._config.rest_base_url,
        ]
        for base_url in urls:
            endpoint = f"{base_url.rstrip('/')}/api/v3/exchangeInfo"
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.get(endpoint, params={"symbol": upper_sym})
                    if response.status_code in (418, 429):
                        continue
                    response.raise_for_status()
                    data = response.json()
                    for sym_info in data.get("symbols", []):
                        if sym_info.get("symbol") == upper_sym:
                            for flt in sym_info.get("filters", []):
                                if flt.get("filterType") == "LOT_SIZE":
                                    step = str(flt.get("stepSize", "0.00000100"))
                                    self._step_size_cache[upper_sym] = step
                                    return step
            except httpx.HTTPError:
                continue

        ws_endpoints = (
            "wss://ws-api.binance.com:443/ws-api/v3",
            "wss://ws-api.binance.com:9443/ws-api/v3",
        )
        req_payload = json.dumps(
            {
                "id": f"info-{int(time.time() * 1000)}",
                "method": "exchangeInfo",
                "params": {"symbol": upper_sym},
            }
        )
        for ws_url in ws_endpoints:
            try:
                with ws_sync_connect(
                    ws_url,
                    open_timeout=self._timeout_seconds,
                    close_timeout=5.0,
                ) as ws_conn:
                    ws_conn.send(req_payload)
                    raw_msg = ws_conn.recv(timeout=self._timeout_seconds)
                    decoded = json.loads(raw_msg)
                    if decoded.get("status") == 200:
                        for sym_info in decoded.get("result", {}).get("symbols", []):
                            if sym_info.get("symbol") == upper_sym:
                                for flt in sym_info.get("filters", []):
                                    if flt.get("filterType") == "LOT_SIZE":
                                        step = str(flt.get("stepSize", "0.00000100"))
                                        self._step_size_cache[upper_sym] = step
                                        return step
            except Exception as exc:
                logger.debug("WebSocket exchangeInfo failed on %s: %s", ws_url, exc)

        return "0.00000100"

    def place_market_buy_by_quote(
        self,
        symbol: str,
        quote_amount_usdt: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Execute a live Spot MARKET BUY order spending a specific USDT amount.

        Args:
            symbol: Trading pair symbol.
            quote_amount_usdt: USDT amount to spend on the purchase.
            client_order_id: Deterministic idempotency key for the order.

        Returns:
            Raw JSON execution response from Binance Spot API.
        """
        params = {
            "symbol": symbol.upper(),
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": f"{quote_amount_usdt:.2f}",
            "newClientOrderId": client_order_id[:36],
        }
        res = self._request_with_failover("POST", "/api/v3/order", params)
        return res

    def place_market_sell_by_quantity(
        self,
        symbol: str,
        quantity: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Execute a live Spot MARKET SELL order for a specific base asset quantity.

        Args:
            symbol: Trading pair symbol.
            quantity: Base asset quantity to sell.
            client_order_id: Deterministic idempotency key for the order.

        Returns:
            Raw JSON execution response from Binance Spot API.
        """
        upper_sym = symbol.upper()
        step_size = self.fetch_symbol_step_size(upper_sym)
        formatted_qty = quantize_step_size(quantity, step_size)
        params = {
            "symbol": upper_sym,
            "side": "SELL",
            "type": "MARKET",
            "quantity": formatted_qty,
            "newClientOrderId": client_order_id[:36],
        }
        try:
            res = self._request_with_failover("POST", "/api/v3/order", params)
            return res
        except LiveExchangeExecutionError as exc:
            if upper_sym.endswith("USDT") and "400" in str(exc):
                base_asset = upper_sym[:-4]
                self._balance_cache.pop(base_asset, None)
                available_base = self.fetch_available_asset_balance(base_asset)
                if 0.0 < available_base < quantity:
                    params["quantity"] = quantize_step_size(available_base, step_size)
                    res = self._request_with_failover("POST", "/api/v3/order", params)
                    return res
            raise
