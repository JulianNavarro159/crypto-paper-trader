from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from crypto_paper_trader.config import StrategyConfig
from crypto_paper_trader.domain import Candle, MarketSignal, SignalType
from crypto_paper_trader.validation import (
    candles_to_dataframe,
    validate_candle_dataframe,
)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute vectorized Exponential Moving Average over a price series."""
    return series.ewm(span=span, adjust=False).mean()


def compute_bollinger_bands(
    close_series: pd.Series,
    period: int,
    std_multiplier: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute vectorized Bollinger Bands (lower, middle, upper)."""
    middle_band = close_series.rolling(window=period, min_periods=1).mean()
    rolling_std = (
        close_series.rolling(window=period, min_periods=1).std(ddof=0).fillna(0.0)
    )
    lower_band = middle_band - (std_multiplier * rolling_std)
    upper_band = middle_band + (std_multiplier * rolling_std)
    return lower_band, middle_band, upper_band


def compute_rsi(close_series: pd.Series, period: int) -> pd.Series:
    """Compute vectorized Relative Strength Index using Wilder smoothing."""
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = np.where(avg_loss == 0.0, 100.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi)
    return pd.Series(rsi, index=close_series.index, dtype=float).fillna(50.0)


def compute_atr(
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    period: int,
) -> pd.Series:
    """Compute vectorized Average True Range volatility indicator."""
    previous_close = close_series.shift(1)
    true_range_components = pd.concat(
        [
            high_series - low_series,
            (high_series - previous_close).abs(),
            (low_series - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = true_range_components.max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def enrich_with_indicators(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Enrich an OHLCV DataFrame with multi-regime technical indicators and signals."""
    enriched = frame.copy()
    enriched["fast_ema"] = compute_ema(enriched["close_price"], config.fast_ema_period)
    enriched["slow_ema"] = compute_ema(enriched["close_price"], config.slow_ema_period)
    enriched["trend_ema"] = compute_ema(
        enriched["close_price"], config.trend_ema_period
    )
    enriched["rsi"] = compute_rsi(enriched["close_price"], config.rsi_period)
    enriched["atr"] = compute_atr(
        enriched["high_price"],
        enriched["low_price"],
        enriched["close_price"],
        config.atr_period,
    )
    bb_lower, bb_mid, bb_upper = compute_bollinger_bands(
        enriched["close_price"],
        config.bollinger_period,
        config.bollinger_std,
    )
    enriched["bb_lower"] = bb_lower
    enriched["bb_mid"] = bb_mid
    enriched["bb_upper"] = bb_upper
    enriched["vol_sma"] = (
        enriched["volume"].rolling(window=config.bollinger_period, min_periods=1).mean()
    )

    prev_rsi = enriched["rsi"].shift(1)
    prev_low = enriched["low_price"].shift(1)
    prev_vol = enriched["volume"].shift(1)
    prev_vol_sma = enriched["vol_sma"].shift(1)

    volume_climax = (prev_vol >= (prev_vol_sma * config.volume_climax_multiplier)) | (
        enriched["volume"] >= (enriched["vol_sma"] * config.volume_climax_multiplier)
    )

    climax_dip_rebound = (
        (prev_low <= enriched["bb_lower"].shift(1))
        & (prev_rsi <= config.rsi_oversold_threshold)
        & volume_climax
        & (enriched["close_price"] > enriched["open_price"])
        & (enriched["close_price"] > enriched["close_price"].shift(1))
    )

    uptrend_pullback = (
        (enriched["fast_ema"] > enriched["slow_ema"])
        & (enriched["slow_ema"] > enriched["trend_ema"])
        & (enriched["trend_ema"] > enriched["trend_ema"].shift(5))
        & (prev_low <= enriched["slow_ema"].shift(1))
        & (enriched["close_price"] > enriched["fast_ema"])
        & (enriched["close_price"] > enriched["open_price"])
        & (enriched["rsi"] >= config.rsi_entry_min)
        & (enriched["rsi"] <= config.rsi_entry_max)
    )

    buy_mask = climax_dip_rebound | uptrend_pullback
    sell_mask = (enriched["rsi"] >= config.rsi_overbought_exit) | (
        enriched["high_price"] >= enriched["bb_upper"]
    )

    signals = np.where(
        buy_mask,
        SignalType.BUY.value,
        np.where(sell_mask, SignalType.SELL.value, SignalType.HOLD.value),
    )
    enriched["signal_type"] = signals
    return enriched


def build_market_signal_from_row(
    row: pd.Series,
    symbol: str,
    config: StrategyConfig,
) -> MarketSignal:
    """Construct a MarketSignal domain instance from an enriched DataFrame row."""
    reference_price = float(row["close_price"])
    raw_atr = float(row["atr"])
    effective_atr = max(raw_atr, reference_price * 0.0025)

    stop_loss_price = round(
        reference_price - (effective_atr * config.atr_stop_loss_multiplier), 8
    )
    take_profit_price = round(
        reference_price + (effective_atr * config.atr_take_profit_multiplier), 8
    )

    close_time = row["close_time"]
    evaluated_at = (
        close_time.to_pydatetime()
        if hasattr(close_time, "to_pydatetime")
        else close_time
    )

    return MarketSignal(
        symbol=symbol,
        evaluated_at=evaluated_at,
        signal_type=SignalType(str(row["signal_type"])),
        reference_price=reference_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        atr_value=round(effective_atr, 6),
        rsi_value=round(float(row["rsi"]), 4),
        fast_ema=round(float(row["fast_ema"]), 6),
        slow_ema=round(float(row["slow_ema"]), 6),
    )


def evaluate_market_signal(
    candles: Sequence[Candle],
    symbol: str,
    config: StrategyConfig,
) -> MarketSignal:
    """Evaluate a sequence of candles and produce a deterministic trading signal."""
    min_required_rows = (
        max(
            config.slow_ema_period,
            config.rsi_period,
            config.atr_period,
            config.bollinger_period,
        )
        + 2
    )
    frame = candles_to_dataframe(candles)
    validate_candle_dataframe(frame, min_rows=min_required_rows)
    enriched = enrich_with_indicators(frame, config)
    return build_market_signal_from_row(enriched.iloc[-1], symbol, config)
