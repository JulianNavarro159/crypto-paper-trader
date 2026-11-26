from __future__ import annotations

import asyncio
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
    locked_symbols: tuple[str, ...] = ()
    unlocked_symbols: tuple[str, ...] = ()


def evaluate_symbol_stabilization(
    symbol: str,
    daily_candles: Sequence[Candle],
    previous_state: BtcLockState,
    strict: bool = False,
    repaired_streak: int = 2,
) -> tuple[BtcLockState, tuple[str, ...]]:
    """Evaluate daily market structure for any symbol using the 3-day no-drop lock.

    Args:
        symbol: Trading pair identifier such as BTCUSDT or SOLUSDT.
        daily_candles: Chronological daily (1d) candles for the symbol.
        previous_state: Current stabilization lock state before this evaluation.
        strict: Require daily EMA10 >= EMA20 in addition to 3-day low and EMA10.
        repaired_streak: Consecutive evaluations passing repair criteria.

    Returns:
        Tuple of (next BtcLockState, tuple of human-readable Spanish reasons).
    """
    clean_asset = symbol[:-4] if symbol.upper().endswith("USDT") else symbol.upper()
    if len(daily_candles) < 20:
        return (
            BtcLockState.LOCKED,
            (
                f"Datos diarios insuficientes para validar estabilizacion de "
                f"{clean_asset}",
            ),
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
                        f"Estructura diaria reparada en {clean_asset} (3 dias "
                        "sin nuevos minimos y cierre sobre EMA10 diaria confirmado)",
                    ),
                )
            return (
                BtcLockState.LOCKED,
                (
                    f"Posible estabilizacion diaria en {clean_asset} en observacion "
                    "(esperando segunda confirmacion)",
                ),
            )

        reasons: list[str] = []
        if not has_higher_low:
            reasons.append(
                f"{clean_asset} sigue marcando minimos bajos "
                f"({recent_3d_low:,.4f} <= {prior_14d_low:,.4f})"
                if recent_3d_low < 100.0
                else (
                    f"{clean_asset} sigue marcando minimos bajos "
                    f"({recent_3d_low:,.0f} <= {prior_14d_low:,.0f})"
                )
            )
        if not above_ema10:
            reasons.append(
                f"Cierre diario ({last_close:,.4f}) debajo de EMA10 ({ema10:,.4f})"
                if last_close < 100.0
                else (
                    f"Cierre diario ({last_close:,.0f}) debajo de EMA10 "
                    f"({ema10:,.0f})"
                )
            )
        if strict and not ema_aligned:
            reasons.append(
                f"EMA10 diaria ({ema10:,.4f}) aun debajo de EMA20 ({ema20:,.4f})"
                if ema10 < 100.0
                else (
                    f"EMA10 diaria ({ema10:,.0f}) aun debajo de EMA20 "
                    f"({ema20:,.0f})"
                )
            )
        return (BtcLockState.LOCKED, tuple(reasons))

    below_ema10_2pct = last_close < (ema10 * 0.98)
    prior_7d_low = min(lows[-8:-1])
    new_7d_low = last_low < prior_7d_low
    if below_ema10_2pct or new_7d_low:
        relock_reasons: list[str] = []
        if below_ema10_2pct:
            relock_reasons.append(
                f"{clean_asset} cayo mas de 2% bajo EMA10 diaria "
                f"({last_close:,.4f} < {ema10 * 0.98:,.4f})"
                if last_close < 100.0
                else (
                    f"{clean_asset} cayo mas de 2% bajo EMA10 diaria "
                    f"({last_close:,.0f} < {ema10 * 0.98:,.0f})"
                )
            )
        if new_7d_low:
            relock_reasons.append(
                f"{clean_asset} marco nuevo minimo de 7 dias "
                f"({last_low:,.4f} < {prior_7d_low:,.4f})"
                if last_low < 100.0
                else (
                    f"{clean_asset} marco nuevo minimo de 7 dias "
                    f"({last_low:,.0f} < {prior_7d_low:,.0f})"
                )
            )
        return (BtcLockState.LOCKED, tuple(relock_reasons))

    return (
        BtcLockState.OPEN,
        (
            f"Estructura diaria de {clean_asset} estable "
            "(3 dias sin caidas y sobre EMA10)",
        ),
    )


def evaluate_btc_stabilization(
    daily_candles: Sequence[Candle],
    previous_state: BtcLockState,
    strict: bool = False,
    repaired_streak: int = 2,
) -> tuple[BtcLockState, tuple[str, ...]]:
    """Evaluate BTCUSDT daily market structure using the 3-day stabilization lock.

    Args:
        daily_candles: Chronological daily (1d) BTCUSDT candles.
        previous_state: Current BTC lock state before this evaluation.
        strict: Require daily EMA10 >= EMA20 in addition to higher low and EMA10.
        repaired_streak: Number of consecutive evaluations passing repair criteria.

    Returns:
        Tuple of (next BtcLockState, tuple of human-readable Spanish reasons).
    """
    return evaluate_symbol_stabilization(
        symbol="BTCUSDT",
        daily_candles=daily_candles,
        previous_state=previous_state,
        strict=strict,
        repaired_streak=repaired_streak,
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
    total_tracked = len(report.locked_symbols) + len(report.unlocked_symbols)
    universe_lock_segment = ""
    if total_tracked > 1:
        open_tags = ",".join(
            sym[:-4] if sym.endswith("USDT") else sym
            for sym in report.unlocked_symbols
        )
        open_display = open_tags if open_tags else "NONE"
        universe_lock_segment = (
            f" open_3d_lock={len(report.unlocked_symbols)}/{total_tracked}"
            f"[{open_display}]"
        )
    return (
        f"btc_lock={report.btc_lock_state.value} risk={report.risk_level.value} "
        f"btc_price={report.btc_price:.2f} ema10={report.btc_daily_ema10:.2f} "
        f"ema20={report.btc_daily_ema20:.2f} btc_24h={report.btc_change_24h_pct:+.2f}% "
        f"fear_greed={fg_str} global_mcap_24h={mcap_str} "
        f"breadth={report.universe_breadth_pct:.1f}%{universe_lock_segment} | "
        f"{reason_text}"
    )


class MarketIntelligenceService:
    """Asynchronous researcher evaluating 3-day daily structure across symbols.

    Args:
        exchange_config: Exchange connectivity configuration for Binance daily klines.
        mode: Manual or automatic override (AUTO, FORCE_LOCK, FORCE_OPEN).
        strict: Enforce strict EMA10 >= EMA20 alignment on daily structure.
        cache_ttl_seconds: Minimum interval between external HTTP API refreshes.
        tracked_symbols: Universe trading pairs evaluated under the 3-day lock.
    """

    def __init__(
        self,
        exchange_config: ExchangeConfig,
        mode: str = "AUTO",
        strict: bool = False,
        cache_ttl_seconds: float = 3600.0,
        daily_candles_fetcher: Callable[[], Awaitable[list[Candle]]] | None = None,
        fear_greed_fetcher: Callable[[], Awaitable[int | None]] | None = None,
        global_mcap_fetcher: Callable[[], Awaitable[float | None]] | None = None,
        tracked_symbols: Sequence[str] = ("BTCUSDT",),
        daily_candles_by_symbol_fetcher: (
            Callable[[str], Awaitable[list[Candle]]] | None
        ) = None,
        daily_cache_ttl_seconds: float = 540.0,
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
        self._daily_cache_ttl_seconds = daily_cache_ttl_seconds
        self._daily_candles_fetcher = daily_candles_fetcher
        self._daily_candles_by_symbol_fetcher = daily_candles_by_symbol_fetcher
        self._fear_greed_fetcher = fear_greed_fetcher
        self._global_mcap_fetcher = global_mcap_fetcher
        unique_syms = tuple(
            dict.fromkeys(["BTCUSDT", *(sym.upper() for sym in tracked_symbols)])
        )
        self._tracked_symbols: tuple[str, ...] = unique_syms
        self._lock_state_by_symbol: dict[str, BtcLockState] = {
            sym: BtcLockState.LOCKED for sym in self._tracked_symbols
        }
        self._repaired_streak_by_symbol: dict[str, int] = {
            sym: 0 for sym in self._tracked_symbols
        }
        self._cached_daily_by_symbol: dict[str, tuple[float, list[Candle]]] = {}
        self._current_lock_state: BtcLockState = BtcLockState.LOCKED
        self._repaired_streak: int = 0
        self._last_report: MarketIntelligenceReport | None = None
        self._last_external_fetch_ts: float = 0.0
        self._coingecko_backoff_until: float = 0.0
        self._cached_fg: int | None = None
        self._cached_mcap_chg: float | None = None

    @property
    def last_report(self) -> MarketIntelligenceReport | None:
        """Return the most recently computed MarketIntelligenceReport, if any."""
        return self._last_report

    async def _fetch_symbol_daily_candles(
        self, symbol: str, force: bool = False
    ) -> list[Candle]:
        """Fetch recent daily candles for symbol using cache and multi-cluster REST."""
        upper_sym = symbol.upper()
        if upper_sym == "BTCUSDT" and self._daily_candles_fetcher is not None:
            return await self._daily_candles_fetcher()
        if self._daily_candles_by_symbol_fetcher is not None:
            return await self._daily_candles_by_symbol_fetcher(upper_sym)

        now_mono = time.monotonic()
        cached = self._cached_daily_by_symbol.get(upper_sym)
        if (
            not force
            and cached is not None
            and (now_mono - cached[0]) < self._daily_cache_ttl_seconds
        ):
            return cached[1]

        if BinanceMarketDataClient.is_cooldown_active():
            return cached[1] if cached is not None else []

        daily_cfg = replace(self._exchange_config, symbol=upper_sym, interval="1d")
        client = BinanceMarketDataClient(config=daily_cfg)
        raw_rows = await client._fetch_klines_with_failover(
            {"symbol": upper_sym, "interval": "1d", "limit": 35}
        )
        parsed = [parse_rest_kline_row(upper_sym, row) for row in raw_rows]
        if parsed:
            self._cached_daily_by_symbol[upper_sym] = (now_mono, parsed)
        return parsed

    async def _fetch_btc_daily_candles(self) -> list[Candle]:
        """Fetch recent BTCUSDT daily candles using Binance multi-cluster failover."""
        return await self._fetch_symbol_daily_candles("BTCUSDT", force=True)

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
        """Fetch 24h global crypto market cap change from CoinGecko with 429 backoff."""
        if self._global_mcap_fetcher is not None:
            return await self._global_mcap_fetcher()

        now_mono = time.monotonic()
        if now_mono < self._coingecko_backoff_until:
            return None

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get("https://api.coingecko.com/api/v3/global")
                if resp.status_code == 429:
                    self._coingecko_backoff_until = now_mono + 7200.0
                    logger.info(
                        "CoinGecko HTTP 429 encountered; pausing global market "
                        "cap refresh for 2h and retaining cached value."
                    )
                    return None
                resp.raise_for_status()
                payload = resp.json()
                val = payload["data"]["market_cap_change_percentage_24h_usd"]
                return round(float(val), 2)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("CoinGecko global market fetch failed: %s", exc)
        return None

    def _update_symbol_lock_state(
        self,
        symbol: str,
        daily_candles: Sequence[Candle],
    ) -> tuple[BtcLockState, tuple[str, ...]]:
        """Advance hysteresis lock state for a single tracked symbol."""
        upper_sym = symbol.upper()
        prev_state = self._lock_state_by_symbol.get(upper_sym, BtcLockState.LOCKED)
        prev_streak = self._repaired_streak_by_symbol.get(upper_sym, 0)

        if self._mode == "FORCE_OPEN":
            self._lock_state_by_symbol[upper_sym] = BtcLockState.OPEN
            return (
                BtcLockState.OPEN,
                ("Desbloqueo manual activo (BTC_BUY_LOCK_MODE=FORCE_OPEN)",),
            )
        if upper_sym == "BTCUSDT" and self._mode == "FORCE_LOCK":
            self._lock_state_by_symbol[upper_sym] = BtcLockState.LOCKED
            self._repaired_streak_by_symbol[upper_sym] = 0
            return (
                BtcLockState.LOCKED,
                ("Bloqueo manual activo (BTC_BUY_LOCK_MODE=FORCE_LOCK)",),
            )
        if len(daily_candles) < 20:
            self._lock_state_by_symbol[upper_sym] = prev_state
            return (
                prev_state,
                ("Sin velas diarias suficientes; manteniendo estado previo",),
            )

        if prev_state == BtcLockState.LOCKED:
            probe_state, _ = evaluate_symbol_stabilization(
                symbol=upper_sym,
                daily_candles=daily_candles,
                previous_state=BtcLockState.LOCKED,
                strict=self._strict,
                repaired_streak=2,
            )
            next_streak = prev_streak + 1 if probe_state == BtcLockState.OPEN else 0
            self._repaired_streak_by_symbol[upper_sym] = next_streak
            next_state, sym_reasons = evaluate_symbol_stabilization(
                symbol=upper_sym,
                daily_candles=daily_candles,
                previous_state=BtcLockState.LOCKED,
                strict=self._strict,
                repaired_streak=next_streak,
            )
            self._lock_state_by_symbol[upper_sym] = next_state
            return next_state, sym_reasons

        next_state, sym_reasons = evaluate_symbol_stabilization(
            symbol=upper_sym,
            daily_candles=daily_candles,
            previous_state=BtcLockState.OPEN,
            strict=self._strict,
            repaired_streak=2,
        )
        self._lock_state_by_symbol[upper_sym] = next_state
        if next_state == BtcLockState.LOCKED:
            self._repaired_streak_by_symbol[upper_sym] = 0
        return next_state, sym_reasons

    async def refresh(
        self,
        candles_by_symbol: dict[str, Sequence[Candle]] | None = None,
        force: bool = False,
    ) -> MarketIntelligenceReport:
        """Research live market conditions and update 3-day lock across symbols.

        Args:
            candles_by_symbol: Optional buffered universe candles for breadth metrics.
            force: Bypass external API cache TTL when True.

        Returns:
            Updated MarketIntelligenceReport with per-symbol lock state and sentiment.
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

        btc_daily_candles: list[Candle] = []
        btc_reasons: tuple[str, ...] = (
            "Sin velas diarias suficientes; manteniendo estado previo",
        )
        locked_list: list[str] = []
        unlocked_list: list[str] = []

        for idx, sym in enumerate(self._tracked_symbols):
            sym_daily: list[Candle] = []
            try:
                sym_daily = await self._fetch_symbol_daily_candles(sym, force=force)
            except Exception as exc:
                logger.warning(
                    "Daily candles fetch failed for %s (%s); preserving lock state.",
                    sym,
                    exc,
                )
            if sym == "BTCUSDT":
                btc_daily_candles = sym_daily
            sym_state, sym_reasons = self._update_symbol_lock_state(sym, sym_daily)
            if sym == "BTCUSDT":
                btc_reasons = sym_reasons
                self._current_lock_state = sym_state
                self._repaired_streak = self._repaired_streak_by_symbol.get(
                    "BTCUSDT", 0
                )
            if sym_state == BtcLockState.LOCKED:
                locked_list.append(sym)
            else:
                unlocked_list.append(sym)
            if (
                idx + 1 < len(self._tracked_symbols)
                and self._daily_candles_by_symbol_fetcher is None
                and self._daily_candles_fetcher is None
            ):
                await asyncio.sleep(0.20)

        btc_price = 0.0
        ema10_val = 0.0
        ema20_val = 0.0
        if len(btc_daily_candles) >= 2:
            ordered_daily = sorted(btc_daily_candles, key=lambda c: c.open_time)
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
        if len(btc_daily_candles) < 20 and self._last_report is not None:
            ema10_val = ema10_val or self._last_report.btc_daily_ema10
            ema20_val = ema20_val or self._last_report.btc_daily_ema20

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
            reasons=btc_reasons,
            locked_symbols=tuple(sorted(locked_list)),
            unlocked_symbols=tuple(sorted(unlocked_list)),
        )
        self._last_report = report
        return report
