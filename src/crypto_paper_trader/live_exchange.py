from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Sequence
from urllib.parse import urlencode

import httpx

from crypto_paper_trader.config import ExchangeConfig

logger = logging.getLogger(__name__)


class LiveExchangeExecutionError(RuntimeError):
    """Exception raised when an authenticated Binance Spot API order fails."""


def quantize_step_size(quantity: float, step_size: str) -> str:
    """Round a base asset quantity down to the exact Binance LOT_SIZE step."""
    qty_dec = Decimal(str(quantity))
    step_dec = Decimal(step_size)
    if step_dec <= Decimal("0"):
        return f"{quantity:.8f}"
    quantized = (qty_dec // step_dec) * step_dec
    return format(quantized.quantize(step_dec, rounding=ROUND_DOWN), "f")


class BinanceSpotAuthenticatedClient:
    """Authenticated Binance Spot REST client for real-money autonomous trading."""

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

    @classmethod
    def from_env(cls, config: ExchangeConfig) -> BinanceSpotAuthenticatedClient:
        """Instantiate an authenticated Binance client using environment variables."""
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        if not api_key or not api_secret:
            raise LiveExchangeExecutionError(
                "Missing BINANCE_API_KEY or BINANCE_API_SECRET in environment."
            )
        return cls(config=config, api_key=api_key, api_secret=api_secret)

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Append timestamp and HMAC-SHA256 signature to request parameters."""
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

    def fetch_available_asset_balance(self, asset: str) -> float:
        """Query the user real Binance Spot account for free balance of an asset."""
        endpoint = f"{self._config.rest_base_url}/api/v3/account"
        params = self._sign_params({})
        upper_asset = asset.upper()
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.get(endpoint, params=params, headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise LiveExchangeExecutionError(
                f"Failed to fetch Binance Spot account balance: {exc}"
            ) from exc

        for asset_entry in payload.get("balances", []):
            if asset_entry.get("asset") == upper_asset:
                return float(asset_entry.get("free", 0.0))
        return 0.0

    def fetch_available_usdt_balance(self) -> float:
        """Query the user real Binance Spot account for free USDT balance."""
        return self.fetch_available_asset_balance("USDT")

    def fetch_active_spot_holding(
        self,
        universe_symbols: Sequence[str],
    ) -> tuple[str, float, float] | None:
        """Detect an existing non-USDT holding in the user Spot account."""
        account_endpoint = f"{self._config.rest_base_url}/api/v3/account"
        params = self._sign_params({})
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.get(
                    account_endpoint, params=params, headers=self._headers()
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError:
            return None

        balances_by_asset = {
            str(entry.get("asset", "")).upper(): float(entry.get("free", 0.0))
            for entry in payload.get("balances", [])
        }

        for sym in universe_symbols:
            upper_sym = sym.upper()
            if not upper_sym.endswith("USDT"):
                continue
            base_asset = upper_sym[:-4]
            qty = balances_by_asset.get(base_asset, 0.0)
            if qty <= 0.0:
                continue

            trades_endpoint = f"{self._config.rest_base_url}/api/v3/myTrades"
            trade_params = self._sign_params({"symbol": upper_sym, "limit": 5})
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    trade_resp = client.get(
                        trades_endpoint,
                        params=trade_params,
                        headers=self._headers(),
                    )
                    trade_resp.raise_for_status()
                    trades = trade_resp.json()
            except httpx.HTTPError:
                continue

            if isinstance(trades, list) and trades:
                last_buy = next(
                    (t for t in reversed(trades) if t.get("isBuyer")),
                    trades[-1],
                )
                entry_price = float(last_buy.get("price", 0.0))
                if entry_price > 0.0 and (qty * entry_price) >= 5.0:
                    return (upper_sym, qty, entry_price)
        return None

    def fetch_symbol_step_size(self, symbol: str) -> str:
        """Fetch LOT_SIZE stepSize for a symbol from Binance exchangeInfo."""
        upper_sym = symbol.upper()
        if upper_sym in self._step_size_cache:
            return self._step_size_cache[upper_sym]

        endpoint = f"{self._config.rest_base_url}/api/v3/exchangeInfo"
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.get(endpoint, params={"symbol": upper_sym})
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LiveExchangeExecutionError(
                f"Failed to fetch exchangeInfo for {upper_sym}: {exc}"
            ) from exc

        for sym_info in data.get("symbols", []):
            if sym_info.get("symbol") == upper_sym:
                for flt in sym_info.get("filters", []):
                    if flt.get("filterType") == "LOT_SIZE":
                        step = str(flt.get("stepSize", "0.00000100"))
                        self._step_size_cache[upper_sym] = step
                        return step

        return "0.00000100"

    def place_market_buy_by_quote(
        self,
        symbol: str,
        quote_amount_usdt: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Execute a live Spot MARKET BUY order spending a specific USDT amount."""
        endpoint = f"{self._config.rest_base_url}/api/v3/order"
        params = self._sign_params(
            {
                "symbol": symbol.upper(),
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{quote_amount_usdt:.2f}",
                "newClientOrderId": client_order_id[:36],
            }
        )
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(endpoint, params=params, headers=self._headers())
                response.raise_for_status()
                return dict(response.json())
        except httpx.HTTPError as exc:
            raise LiveExchangeExecutionError(
                f"Live Binance MARKET BUY failed for {symbol}: {exc}"
            ) from exc

    def place_market_sell_by_quantity(
        self,
        symbol: str,
        quantity: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        """Execute a live Spot MARKET SELL order for a specific base asset quantity."""
        upper_sym = symbol.upper()
        effective_qty = quantity
        if upper_sym.endswith("USDT"):
            base_asset = upper_sym[:-4]
            available_base = self.fetch_available_asset_balance(base_asset)
            if 0.0 < available_base < effective_qty:
                effective_qty = available_base

        step_size = self.fetch_symbol_step_size(upper_sym)
        formatted_qty = quantize_step_size(effective_qty, step_size)
        endpoint = f"{self._config.rest_base_url}/api/v3/order"
        params = self._sign_params(
            {
                "symbol": upper_sym,
                "side": "SELL",
                "type": "MARKET",
                "quantity": formatted_qty,
                "newClientOrderId": client_order_id[:36],
            }
        )
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(endpoint, params=params, headers=self._headers())
                response.raise_for_status()
                return dict(response.json())
        except httpx.HTTPError as exc:
            raise LiveExchangeExecutionError(
                f"Live Binance MARKET SELL failed for {symbol}: {exc}"
            ) from exc
