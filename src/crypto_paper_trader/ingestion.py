from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence

import httpx
import websockets

from crypto_paper_trader.config import ExchangeConfig
from crypto_paper_trader.domain import Candle

logger = logging.getLogger(__name__)


class MarketDataIngestionError(RuntimeError):
    """Exception raised when fetching or streaming market data fails."""


class MarketDataProvider(Protocol):
    """Protocol defining asynchronous market data ingestion operations."""

    async def fetch_historical_candles(self) -> list[Candle]:
        """Fetch bootstrap historical OHLCV candles from the exchange."""

    def stream_realtime_candles(
        self, max_events: int | None = None
    ) -> AsyncIterator[Candle]:
        """Stream live OHLCV candle updates from the exchange."""


def parse_rest_kline_row(symbol: str, row: Sequence[Any]) -> Candle:
    """Parse a raw Binance REST API kline array into a Candle domain model.

    Args:
        symbol: Trading pair identifier.
        row: Twelve-element kline array returned by Binance REST endpoint.

    Returns:
        Validated Candle instance with UTC timestamps.

    Raises:
        MarketDataIngestionError: If the payload structure is malformed.
    """
    if len(row) < 7:
        raise MarketDataIngestionError(f"Invalid REST kline payload length: {len(row)}")

    open_time = datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc)
    close_time = datetime.fromtimestamp(int(row[6]) / 1000.0, tz=timezone.utc)

    return Candle(
        symbol=symbol,
        open_time=open_time,
        close_time=close_time,
        open_price=float(row[1]),
        high_price=float(row[2]),
        low_price=float(row[3]),
        close_price=float(row[4]),
        volume=float(row[5]),
        is_closed=True,
    )


def parse_ws_kline_payload(payload: dict[str, Any]) -> Candle:
    """Parse a Binance WebSocket kline event dictionary into a Candle domain model.

    Args:
        payload: Decoded JSON dictionary from the Binance kline stream.

    Returns:
        Validated Candle instance representing the live update.

    Raises:
        MarketDataIngestionError: If required keys are missing from payload.
    """
    if "k" not in payload:
        raise MarketDataIngestionError("WebSocket message missing 'k' kline object.")

    kline = payload["k"]
    open_time = datetime.fromtimestamp(int(kline["t"]) / 1000.0, tz=timezone.utc)
    close_time = datetime.fromtimestamp(int(kline["T"]) / 1000.0, tz=timezone.utc)

    return Candle(
        symbol=str(kline["s"]),
        open_time=open_time,
        close_time=close_time,
        open_price=float(kline["o"]),
        high_price=float(kline["h"]),
        low_price=float(kline["l"]),
        close_price=float(kline["c"]),
        volume=float(kline["v"]),
        is_closed=bool(kline["x"]),
    )


class BinanceMarketDataClient:
    """Binance Spot public market data client using REST and WebSockets.

    Args:
        config: Exchange connectivity and symbol configuration.
        http_timeout_seconds: Timeout in seconds for REST requests.
    """

    def __init__(
        self,
        config: ExchangeConfig,
        http_timeout_seconds: float = 10.0,
    ) -> None:
        self._config = config
        self._http_timeout_seconds = http_timeout_seconds

    async def _fetch_klines_with_failover(
        self, params: dict[str, str | int]
    ) -> list[Any]:
        """Fetch klines with automatic multi-cluster failover on HTTP 418 or 429.

        Args:
            params: Query parameters for the klines endpoint.

        Returns:
            List of raw kline records from Binance.

        Raises:
            MarketDataIngestionError: If all cluster endpoints fail.
        """
        urls = [
            self._config.market_data_url,
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com",
            self._config.rest_base_url,
        ]
        last_error: Exception | None = None
        for url in urls:
            endpoint = f"{url.rstrip('/')}/api/v3/klines"
            try:
                async with httpx.AsyncClient(
                    timeout=self._http_timeout_seconds,
                    headers={"User-Agent": "CryptoPaperTrader/1.0"},
                ) as http_client:
                    response = await http_client.get(endpoint, params=params)
                    if response.status_code == 418:
                        last_error = httpx.HTTPStatusError(
                            f"Rate limited: HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                        logger.warning(
                            "Binance IP limit 418 on %s; cooling down 60.0s...",
                            url,
                        )
                        await asyncio.sleep(60.0)
                        break
                    if response.status_code == 429:
                        last_error = httpx.HTTPStatusError(
                            f"Rate limited: HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                        logger.warning(
                            "Binance rate limit 429 on %s; trying next cluster...",
                            url,
                        )
                        await asyncio.sleep(2.0)
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, list):
                        return payload
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Kline fetch failed on %s (%s); trying fallback cluster...",
                    url,
                    exc,
                )
                await asyncio.sleep(0.5)

        raise MarketDataIngestionError(
            f"All Binance market data clusters failed: {last_error}"
        )

    async def fetch_historical_candles(self) -> list[Candle]:
        """Fetch historical OHLCV candles from Binance Spot public REST API.

        Returns:
            Chronological list of closed historical Candle instances.

        Raises:
            MarketDataIngestionError: If the HTTP request fails or payload is invalid.
        """
        params: dict[str, str | int] = {
            "symbol": self._config.symbol.upper(),
            "interval": self._config.interval,
            "limit": self._config.bootstrap_candle_limit,
        }
        raw_rows = await self._fetch_klines_with_failover(params)
        return [
            parse_rest_kline_row(self._config.symbol.upper(), row) for row in raw_rows
        ]

    async def fetch_paginated_historical_candles(
        self,
        total_candles: int = 1000,
    ) -> list[Candle]:
        """Fetch multiple pages of historical OHLCV candles from Binance REST API.

        Args:
            total_candles: Total count of historical candles to retrieve.

        Returns:
            Chronological deduplicated list of historical Candle instances.

        Raises:
            MarketDataIngestionError: If any HTTP request fails or payload is invalid.
        """
        symbol = self._config.symbol.upper()
        collected: dict[int, Candle] = {}
        end_time_ms: int | None = None
        remaining = max(total_candles, 1)

        while remaining > 0:
            batch_limit = min(remaining, 1000)
            params: dict[str, str | int] = {
                "symbol": symbol,
                "interval": self._config.interval,
                "limit": batch_limit,
            }
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

            raw_rows = await self._fetch_klines_with_failover(params)
            if not raw_rows:
                break

            batch_candles = [parse_rest_kline_row(symbol, row) for row in raw_rows]
            for candle in batch_candles:
                collected[int(candle.open_time.timestamp())] = candle

            earliest_open_ms = int(raw_rows[0][0])
            end_time_ms = earliest_open_ms - 1
            remaining -= len(batch_candles)

            if len(batch_candles) < batch_limit:
                break

        return [collected[key] for key in sorted(collected.keys())]

    async def stream_realtime_candles(
        self, max_events: int | None = None
    ) -> AsyncIterator[Candle]:
        """Stream real-time kline ticks from Binance Spot with auto-reconnect.

        Args:
            max_events: Optional maximum number of WebSocket messages to yield.

        Yields:
            Candle instances updated in real time.
        """
        stream_name = f"{self._config.symbol.lower()}@kline_{self._config.interval}"
        ws_url = f"{self._config.ws_base_url}/{stream_name}"
        events_received = 0

        while True:
            try:
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=20
                ) as ws_connection:
                    async for raw_message in ws_connection:
                        payload = json.loads(raw_message)
                        candle = parse_ws_kline_payload(payload)
                        events_received += 1
                        yield candle
                        if max_events is not None and events_received >= max_events:
                            return
            except (
                websockets.WebSocketException,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                logger.warning(
                    "WebSocket disconnected (%s); reconnecting in 3 seconds...",
                    exc,
                )
                await asyncio.sleep(3.0)
