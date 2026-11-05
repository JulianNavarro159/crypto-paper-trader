from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from crypto_paper_trader.config import StrategyConfig
from crypto_paper_trader.domain import (
    Candle,
    MacroMarketRegime,
    MarketSignal,
    SignalType,
)
from crypto_paper_trader.validation import (
    candles_to_dataframe,
    validate_candle_dataframe,
)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute vectorized Exponential Moving Average over a price series.

    Args:
        series: Input numeric price series.
        span: Lookback span in periods.

    Returns:
        Series containing the exponential moving average values.
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_bollinger_bands(
    close_series: pd.Series,
    period: int,
    std_multiplier: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute vectorized Bollinger Bands (lower, middle, upper).

    Args:
        close_series: Series of closing prices.
        period: Moving average and standard deviation lookback window.
        std_multiplier: Multiplier applied to rolling standard deviation.

    Returns:
        Tuple of (lower_band, middle_band, upper_band) Series.
    """
    middle_band = close_series.rolling(window=period, min_periods=1).mean()
    rolling_std = (
        close_series.rolling(window=period, min_periods=1).std(ddof=0).fillna(0.0)
    )
    lower_band = middle_band - (std_multiplier * rolling_std)
    upper_band = middle_band + (std_multiplier * rolling_std)
    return lower_band, middle_band, upper_band


def compute_rsi(close_series: pd.Series, period: int) -> pd.Series:
    """Compute vectorized Relative Strength Index using Wilder smoothing.

    Args:
        close_series: Series of closing prices.
        period: Lookback window for RSI calculation.

    Returns:
        Series of RSI values bounded between 0 and 100.
    """
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
    """Compute vectorized Average True Range volatility indicator.

    Args:
        high_series: Series of high prices.
        low_series: Series of low prices.
        close_series: Series of close prices.
        period: Lookback window for smoothing true range.

    Returns:
        Series of strictly positive ATR values.
    """
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
    """Enrich an OHLCV DataFrame with multi-regime technical indicators and signals.

    Args:
        frame: Validated DataFrame containing OHLCV columns.
        config: Strategy hyperparameters.

    Returns:
        New DataFrame with trend, momentum, volatility, and signal_type columns.
    """
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

    trend_momentum_breakout = (
        (enriched["fast_ema"] > enriched["slow_ema"])
        & (enriched["close_price"] > enriched["trend_ema"])
        & (
            (enriched["fast_ema"] - enriched["slow_ema"])
            >= (enriched["fast_ema"] - enriched["slow_ema"]).shift(1)
        )
        & (enriched["close_price"] > enriched["high_price"].shift(1))
        & (enriched["close_price"] > enriched["open_price"])
        & (enriched["close_price"] <= (enriched["bb_upper"] * 0.995))
        & (
            (prev_vol >= (prev_vol_sma * 1.05))
            | (enriched["volume"] >= (enriched["vol_sma"] * 1.05))
        )
        & (enriched["rsi"] >= 46.0)
        & (enriched["rsi"] <= 54.0)
    )

    micro_dip_rebound = (
        (prev_low <= enriched["bb_lower"].shift(1) * 1.003)
        & (prev_rsi <= 38.0)
        & (enriched["rsi"] > prev_rsi)
        & (enriched["rsi"] <= 52.0)
        & (enriched["close_price"] > enriched["open_price"])
        & (enriched["close_price"] > enriched["close_price"].shift(1))
        & (
            (enriched["close_price"] - enriched["open_price"])
            >= (enriched["atr"] * 0.12)
        )
    )

    ema_recovery = (
        (enriched["fast_ema"] > enriched["slow_ema"])
        & (enriched["fast_ema"].shift(1) <= enriched["slow_ema"].shift(1))
        & (enriched["close_price"] <= enriched["bb_mid"] * 1.005)
        & (enriched["rsi"] >= 34.0)
        & (enriched["rsi"] <= 53.0)
    )

    rolling_range_high = (
        enriched["high_price"].shift(1).rolling(window=6, min_periods=3).max()
    )
    rolling_range_low = (
        enriched["low_price"].shift(1).rolling(window=6, min_periods=3).min()
    )
    range_compression = (rolling_range_high - rolling_range_low) <= (
        enriched["atr"] * 3.2
    )
    opening_range_breakout = (
        range_compression
        & (enriched["close_price"] > rolling_range_high)
        & (enriched["close_price"] > enriched["open_price"])
        & (enriched["fast_ema"] >= enriched["slow_ema"])
        & (enriched["volume"] >= (enriched["vol_sma"] * 0.95))
        & (enriched["rsi"] >= 45.0)
        & (enriched["rsi"] <= config.rsi_entry_max)
        & (enriched["close_price"] <= (enriched["bb_upper"] * 0.995))
    )

    sell_mask = (
        (enriched["rsi"] >= config.rsi_overbought_exit)
        | (
            (enriched["high_price"] >= enriched["bb_upper"] * 0.998)
            & (enriched["rsi"] >= 58.5)
        )
        | (
            (enriched["rsi"] >= 53.0)
            & (enriched["rsi"] < prev_rsi)
            & (enriched["close_price"] < enriched["open_price"])
        )
    )

    raw_buy_mask = (
        climax_dip_rebound
        | uptrend_pullback
        | trend_momentum_breakout
        | micro_dip_rebound
        | ema_recovery
        | opening_range_breakout
    )
    buy_mask = raw_buy_mask & (~sell_mask)

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
    """Construct a MarketSignal domain instance from an enriched DataFrame row.

    Args:
        row: Series from an indicator-enriched OHLCV DataFrame.
        symbol: Trading pair identifier.
        config: Strategy configuration parameters.

    Returns:
        MarketSignal with dynamic ATR stop loss and take profit levels.
    """
    reference_price = float(row["close_price"])
    raw_atr = float(row["atr"])
    effective_atr = max(raw_atr, reference_price * 0.0015)

    stop_loss_dist = min(
        max(
            effective_atr * config.atr_stop_loss_multiplier,
            reference_price * 0.0040,
        ),
        reference_price * 0.0050,
    )
    take_profit_dist = min(
        max(
            effective_atr * config.atr_take_profit_multiplier,
            reference_price * 0.0075,
        ),
        reference_price * 0.0085,
    )
    stop_loss_price = round(reference_price - stop_loss_dist, 10)
    take_profit_price = round(reference_price + take_profit_dist, 10)

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
        atr_value=round(effective_atr, 10),
        rsi_value=round(float(row["rsi"]), 4),
        fast_ema=round(float(row["fast_ema"]), 10),
        slow_ema=round(float(row["slow_ema"]), 10),
    )


def evaluate_market_signal(
    candles: Sequence[Candle],
    symbol: str,
    config: StrategyConfig,
) -> MarketSignal:
    """Evaluate a sequence of candles and produce a deterministic trading signal.

    Args:
        candles: Historical and latest market candles in chronological order.
        symbol: Trading pair identifier.
        config: Strategy configuration parameters.

    Returns:
        MarketSignal containing the decision and protective price levels.

    Raises:
        MarketDataValidationError: If input candles fail validation checks.
    """
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


def evaluate_macro_regime(
    candles: Sequence[Candle],
    config: StrategyConfig,
) -> MacroMarketRegime:
    """Classify the broad crypto market regime using benchmark price action.

    Args:
        candles: Sequence of benchmark candles such as BTCUSDT.
        config: Strategy hyperparameters for EMA and RSI calculation.

    Returns:
        MacroMarketRegime indicating BULLISH, NEUTRAL, or BEARISH condition.
    """
    min_required_rows = max(config.slow_ema_period, config.rsi_period) + 2
    if len(candles) < min_required_rows:
        return MacroMarketRegime.NEUTRAL

    frame = candles_to_dataframe(candles)
    fast_ema = compute_ema(frame["close_price"], config.fast_ema_period)
    slow_ema = compute_ema(frame["close_price"], config.slow_ema_period)
    rsi = compute_rsi(frame["close_price"], config.rsi_period)

    last_fast = float(fast_ema.iloc[-1])
    last_slow = float(slow_ema.iloc[-1])
    last_rsi = float(rsi.iloc[-1])

    if last_fast < last_slow and last_rsi < 48.0:
        return MacroMarketRegime.BEARISH

    if last_fast > last_slow and last_rsi >= 50.0:
        return MacroMarketRegime.BULLISH

    return MacroMarketRegime.NEUTRAL
