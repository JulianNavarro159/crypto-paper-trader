from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Sequence

import httpx
import pandas as pd

from crypto_paper_trader.config import ExchangeConfig
from crypto_paper_trader.domain import Candle
from crypto_paper_trader.ingestion import BinanceMarketDataClient, parse_rest_kline_row
from crypto_paper_trader.strategy import compute_ema

logger = logging.getLogger(__name__)


class BtcLockState(str, Enum):
    """Operational state controlling whether new BTCUSDT purchases are allowed."""

    LOCKED = "LOCKED"
    OPEN = "OPEN"


class MarketRiskLevel(str, Enum):
    """Broad cryptocurrency market sentiment and breadth classification."""

    PANIC = "PANIC"
    CAUTION = "CAUTION"
    NEUTRAL = "NEUTRAL"
    CONSTRUCTIVE = "CONSTRUCTIVE"


@dataclass(frozen=True)
class MarketIntelligenceReport:
    """Aggregated crypto market research report and BTC stabilization lock state.

    Args:
        evaluated_at: UTC timestamp of the market research cycle.
        btc_lock_state: Whether new BTCUSDT purchases are LOCKED or OPEN.
        risk_level: Macro sentiment and universe breadth risk tier.
        btc_price: Latest BTC daily or spot close price in USDT.
        btc_daily_ema10: Daily 10-period EMA of BTCUSDT.
        btc_daily_ema20: Daily 20-period EMA of BTCUSDT.
        btc_change_24h_pct: Approximate 24h percentage change of BTCUSDT.
        fear_greed_value: Alternative.me Fear and Greed score (0-100) or None.
        global_market_cap_change_24h_pct: CoinGecko 24h market cap change or None.
        universe_breadth_pct: Percentage of universe coins trading higher over 24h.
        reasons: Tuple of concise Spanish reasons explaining the BTC lock decision.
    """

    evaluated_at: datetime
    btc_lock_state: BtcLockState
    risk_level: MarketRiskLevel
    btc_price: float
    btc_daily_ema10: float
    btc_daily_ema20: float
    btc_change_24h_pct: float
    fear_greed_value: int | None
    global_market_cap_change_24h_pct: float | None
    universe_breadth_pct: float
    reasons: tuple[str, ...]


def evaluate_btc_stabilization(
    daily_candles: Sequence[Candle],
    previous_state: BtcLockState,
    strict: bool = False,
    repaired_streak: int = 2,
) -> tuple[BtcLockState, tuple[str, ...]]:
    """Evaluate BTCUSDT daily market structure to decide LOCKED vs OPEN.

    Args:
        daily_candles: Chronological daily (1d) BTCUSDT candles.
        previous_state: Current BTC lock state before this evaluation.
        strict: Require daily EMA10 >= EMA20 in addition to higher low and EMA10.
        repaired_streak: Number of consecutive evaluations passing repair criteria.

    Returns:
        Tuple of (next BtcLockState, tuple of human-readable Spanish reasons).
    """
    if len(daily_candles) < 20:
        return (
            BtcLockState.LOCKED,
            ("Datos diarios insuficientes para validar estabilizacion de BTC",),
        )

    ordered = sorted(daily_candles, key=lambda c: c.open_time)
    closes = pd.Series([float(c.close_price) for c in ordered], dtype=float)
    lows = [float(c.low_price) for c in ordered]
    ema10 = float(compute_ema(closes, 10).iloc[-1])
    ema20 = float(compute_ema(closes, 20).iloc[-1])
    last_close = float(closes.iloc[-1])
    last_low = lows[-1]

    recent_3d_low = min(lows[-3:])
    prior_14d_low = min(lows[-17:-3])
    has_higher_low = recent_3d_low > prior_14d_low
    above_ema10 = last_close > ema10
    ema_aligned = (ema10 >= ema20) if strict else True

    if previous_state == BtcLockState.LOCKED:
        if has_higher_low and above_ema10 and ema_aligned:
            if repaired_streak >= 2:
                return (
                    BtcLockState.OPEN,
                    (
                        "Estructura diaria reparada (minimo ascendente y "
                        "cierre sobre EMA10 diaria confirmado)",
                    ),
                )
            return (
                BtcLockState.LOCKED,
                (
                    "Posible estabilizacion diaria en observacion "
                    "(esperando segunda confirmacion)",
                ),
            )

        reasons: list[str] = []
        if not has_higher_low:
            reasons.append(
                f"BTC sigue marcando minimos bajos ({recent_3d_low:,.0f} <= "
                f"{prior_14d_low:,.0f})"
            )
        if not above_ema10:
            reasons.append(
                f"Cierre diario ({last_close:,.0f}) debajo de EMA10 ({ema10:,.0f})"
            )
        if strict and not ema_aligned:
            reasons.append(
                f"EMA10 diaria ({ema10:,.0f}) aun debajo de EMA20 ({ema20:,.0f})"
            )
        return (BtcLockState.LOCKED, tuple(reasons))

    below_ema10_2pct = last_close < (ema10 * 0.98)
    prior_7d_low = min(lows[-8:-1])
    new_7d_low = last_low < prior_7d_low
    if below_ema10_2pct or new_7d_low:
        relock_reasons: list[str] = []
        if below_ema10_2pct:
            relock_reasons.append(
                f"BTC cayo mas de 2% bajo EMA10 diaria ({last_close:,.0f} < "
                f"{ema10 * 0.98:,.0f})"
            )
        if new_7d_low:
            relock_reasons.append(
                f"BTC marco nuevo minimo de 7 dias ({last_low:,.0f} < "
                f"{prior_7d_low:,.0f})"
            )
        return (BtcLockState.LOCKED, tuple(relock_reasons))

    return (
        BtcLockState.OPEN,
        ("Estructura diaria de BTC estable sobre soporte y EMA10",),
    )


def classify_market_risk(
    fear_greed_value: int | None,
    universe_breadth_pct: float,
) -> MarketRiskLevel:
    """Classify overall crypto market risk from Fear & Greed and universe breadth.

    Args:
        fear_greed_value: Alternative.me Fear & Greed value (0-100) or None.
        universe_breadth_pct: Percentage of universe assets positive over 24h.

    Returns:
        MarketRiskLevel enum classifying PANIC, CAUTION, NEUTRAL, or CONSTRUCTIVE.
    """
    if (
        fear_greed_value is not None
        and fear_greed_value <= 20
        and universe_breadth_pct <= 20.0
    ):
        return MarketRiskLevel.PANIC

    if (
        fear_greed_value is not None and fear_greed_value <= 35
    ) or universe_breadth_pct <= 40.0:
        return MarketRiskLevel.CAUTION

    if (
        fear_greed_value is not None
        and fear_greed_value >= 55
        and universe_breadth_pct >= 60.0
    ):
        return MarketRiskLevel.CONSTRUCTIVE

    return MarketRiskLevel.NEUTRAL


def compute_universe_breadth_pct(
    candles_by_symbol: dict[str, Sequence[Candle]] | None,
) -> tuple[float, float]:
    """Compute universe breadth (% coins up over 24h) and BTC 24h change %.

    Args:
        candles_by_symbol: Mapping of symbol to chronological historical candles.

    Returns:
        Tuple of (breadth_pct_0_to_100, btc_24h_change_pct).
    """
    if not candles_by_symbol:
        return 0.0, 0.0

    positive_count = 0
    evaluated_count = 0
    btc_change = 0.0

    for sym, candles in candles_by_symbol.items():
        if len(candles) < 2:
            continue
        lookback_idx = max(0, len(candles) - 96)
        start_price = float(candles[lookback_idx].close_price)
        end_price = float(candles[-1].close_price)
        if start_price <= 0.0:
            continue
        pct_change = ((end_price - start_price) / start_price) * 100.0
        evaluated_count += 1
        if pct_change > 0.0:
            positive_count += 1
        if sym.upper() == "BTCUSDT":
            btc_change = round(pct_change, 2)

    if evaluated_count == 0:
        return 0.0, btc_change
    breadth = round((positive_count / evaluated_count) * 100.0, 1)
    return breadth, btc_change


def format_market_intelligence_log(report: MarketIntelligenceReport) -> str:
    """Format a compact structured log string summarizing market intelligence.

    Args:
        report: Latest MarketIntelligenceReport instance.

    Returns:
        Single-line summary suitable for application logs.
    """
    fg_str = (
        str(report.fear_greed_value) if report.fear_greed_value is not None else "NA"
    )
    mcap_str = (
        f"{report.global_market_cap_change_24h_pct:+.2f}%"
        if report.global_market_cap_change_24h_pct is not None
        else "NA"
    )
    reason_text = "; ".join(report.reasons)
    return (
        f"btc_lock={report.btc_lock_state.value} risk={report.risk_level.value} "
        f"btc_price={report.btc_price:.2f} ema10={report.btc_daily_ema10:.2f} "
        f"ema20={report.btc_daily_ema20:.2f} btc_24h={report.btc_change_24h_pct:+.2f}% "
        f"fear_greed={fg_str} global_mcap_24h={mcap_str} "
        f"breadth={report.universe_breadth_pct:.1f}% | {reason_text}"
    )


class MarketIntelligenceService:
    """Asynchronous researcher evaluating BTC daily structure and external sentiment.

    Args:
        exchange_config: Exchange connectivity configuration for Binance daily klines.
        mode: Manual or automatic override (AUTO, FORCE_LOCK, FORCE_OPEN).
        strict: Enforce strict EMA10 >= EMA20 alignment on BTC daily structure.
        cache_ttl_seconds: Minimum interval between external HTTP API refreshes.
    """

    def __init__(
        self,
        exchange_config: ExchangeConfig,
        mode: str = "AUTO",
        strict: bool = False,
        cache_ttl_seconds: float = 1800.0,
        daily_candles_fetcher: Callable[[], Awaitable[list[Candle]]] | None = None,
        fear_greed_fetcher: Callable[[], Awaitable[int | None]] | None = None,
        global_mcap_fetcher: Callable[[], Awaitable[float | None]] | None = None,
    ) -> None:
        self._exchange_config = exchange_config
        normalized_mode = mode.strip().upper()
        self._mode = (
            normalized_mode
            if normalized_mode in ("AUTO", "FORCE_LOCK", "FORCE_OPEN")
            else "AUTO"
        )
        self._strict = strict
        self._cache_ttl_seconds = cache_ttl_seconds
        self._daily_candles_fetcher = daily_candles_fetcher
        self._fear_greed_fetcher = fear_greed_fetcher
        self._global_mcap_fetcher = global_mcap_fetcher
        self._current_lock_state: BtcLockState = BtcLockState.LOCKED
        self._repaired_streak: int = 0
        self._last_report: MarketIntelligenceReport | None = None
        self._last_external_fetch_ts: float = 0.0
        self._cached_fg: int | None = None
        self._cached_mcap_chg: float | None = None

    @property
    def last_report(self) -> MarketIntelligenceReport | None:
        """Return the most recently computed MarketIntelligenceReport, if any."""
        return self._last_report

    async def _fetch_btc_daily_candles(self) -> list[Candle]:
        """Fetch recent BTCUSDT daily candles using Binance multi-cluster failover."""
        if self._daily_candles_fetcher is not None:
            return await self._daily_candles_fetcher()

        daily_cfg = replace(self._exchange_config, symbol="BTCUSDT", interval="1d")
        client = BinanceMarketDataClient(config=daily_cfg)
        raw_rows = await client._fetch_klines_with_failover(
            {"symbol": "BTCUSDT", "interval": "1d", "limit": 35}
        )
        return [parse_rest_kline_row("BTCUSDT", row) for row in raw_rows]

    async def _fetch_fear_and_greed(self) -> int | None:
        """Fetch current Fear & Greed Index value from alternative.me."""
        if self._fear_greed_fetcher is not None:
            return await self._fear_greed_fetcher()

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get("https://api.alternative.me/fng/?limit=1")
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data", [])
                if isinstance(data, list) and data:
                    return int(data[0]["value"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Fear & Greed fetch failed: %s", exc)
        return None

    async def _fetch_global_market_cap_change(self) -> float | None:
        """Fetch 24h global crypto market cap percentage change from CoinGecko."""
        if self._global_mcap_fetcher is not None:
            return await self._global_mcap_fetcher()

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get("https://api.coingecko.com/api/v3/global")
                resp.raise_for_status()
                payload = resp.json()
                val = payload["data"]["market_cap_change_percentage_24h_usd"]
                return round(float(val), 2)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("CoinGecko global market fetch failed: %s", exc)
        return None

    async def refresh(
        self,
        candles_by_symbol: dict[str, Sequence[Candle]] | None = None,
        force: bool = False,
    ) -> MarketIntelligenceReport:
        """Research live crypto market conditions and update BTC stabilization state.

        Args:
            candles_by_symbol: Optional buffered universe candles for breadth metrics.
            force: Bypass external API cache TTL when True.

        Returns:
            Updated MarketIntelligenceReport with lock state and sentiment metrics.
        """
        now_dt = datetime.now(tz=timezone.utc)
        now_mono = time.monotonic()

        if (
            force
            or self._last_external_fetch_ts == 0.0
            or (now_mono - self._last_external_fetch_ts) >= self._cache_ttl_seconds
        ):
            fg_val = await self._fetch_fear_and_greed()
            mcap_val = await self._fetch_global_market_cap_change()
            if fg_val is not None:
                self._cached_fg = fg_val
            if mcap_val is not None:
                self._cached_mcap_chg = mcap_val
            self._last_external_fetch_ts = now_mono

        breadth_pct, btc_24h_pct = compute_universe_breadth_pct(candles_by_symbol)
        risk_level = classify_market_risk(self._cached_fg, breadth_pct)

        daily_candles: list[Candle] = []
        try:
            daily_candles = await self._fetch_btc_daily_candles()
        except Exception as exc:
            logger.warning(
                "BTC daily candles fetch failed (%s); preserving lock state %s.",
                exc,
                self._current_lock_state.value,
            )

        btc_price = 0.0
        ema10_val = 0.0
        ema20_val = 0.0
        if len(daily_candles) >= 2:
            ordered_daily = sorted(daily_candles, key=lambda c: c.open_time)
            closes = pd.Series(
                [float(c.close_price) for c in ordered_daily], dtype=float
            )
            btc_price = float(closes.iloc[-1])
            ema10_val = round(float(compute_ema(closes, 10).iloc[-1]), 2)
            ema20_val = round(float(compute_ema(closes, 20).iloc[-1]), 2)
            if btc_24h_pct == 0.0:
                prev_close = float(closes.iloc[-2])
                if prev_close > 0.0:
                    btc_24h_pct = round(
                        ((btc_price - prev_close) / prev_close) * 100.0, 2
                    )
        elif candles_by_symbol and "BTCUSDT" in candles_by_symbol:
            btc_buf = candles_by_symbol["BTCUSDT"]
            if btc_buf:
                btc_price = float(btc_buf[-1].close_price)

        if self._mode == "FORCE_LOCK":
            self._current_lock_state = BtcLockState.LOCKED
            self._repaired_streak = 0
            reasons: tuple[str, ...] = (
                "Bloqueo manual activo (BTC_BUY_LOCK_MODE=FORCE_LOCK)",
            )
        elif self._mode == "FORCE_OPEN":
            self._current_lock_state = BtcLockState.OPEN
            reasons = ("Desbloqueo manual activo (BTC_BUY_LOCK_MODE=FORCE_OPEN)",)
        elif len(daily_candles) < 20:
            if self._last_report is not None:
                self._current_lock_state = self._last_report.btc_lock_state
                ema10_val = ema10_val or self._last_report.btc_daily_ema10
                ema20_val = ema20_val or self._last_report.btc_daily_ema20
            else:
                self._current_lock_state = BtcLockState.LOCKED
            reasons = (
                "Sin velas diarias suficientes; manteniendo estado previo",
            )
        else:
            if self._current_lock_state == BtcLockState.LOCKED:
                probe_state, _ = evaluate_btc_stabilization(
                    daily_candles=daily_candles,
                    previous_state=BtcLockState.LOCKED,
                    strict=self._strict,
                    repaired_streak=2,
                )
                if probe_state == BtcLockState.OPEN:
                    self._repaired_streak += 1
                else:
                    self._repaired_streak = 0
                next_state, reasons = evaluate_btc_stabilization(
                    daily_candles=daily_candles,
                    previous_state=BtcLockState.LOCKED,
                    strict=self._strict,
                    repaired_streak=self._repaired_streak,
                )
                self._current_lock_state = next_state
            else:
                next_state, reasons = evaluate_btc_stabilization(
                    daily_candles=daily_candles,
                    previous_state=BtcLockState.OPEN,
                    strict=self._strict,
                    repaired_streak=2,
                )
                self._current_lock_state = next_state
                if next_state == BtcLockState.LOCKED:
                    self._repaired_streak = 0

        report = MarketIntelligenceReport(
            evaluated_at=now_dt,
            btc_lock_state=self._current_lock_state,
            risk_level=risk_level,
            btc_price=round(btc_price, 2),
            btc_daily_ema10=ema10_val,
            btc_daily_ema20=ema20_val,
            btc_change_24h_pct=btc_24h_pct,
            fear_greed_value=self._cached_fg,
            global_market_cap_change_24h_pct=self._cached_mcap_chg,
            universe_breadth_pct=breadth_pct,
            reasons=reasons,
        )
        self._last_report = report
        return report
