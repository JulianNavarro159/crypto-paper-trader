from __future__ import annotations

from typing import Sequence

import pandas as pd

from crypto_paper_trader.config import ExchangeConfig, RiskConfig
from crypto_paper_trader.domain import Candle, MarketSignal

REQUIRED_CANDLE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
)


class MarketDataValidationError(ValueError):
    """Exception raised when incoming market data fails quality checks."""


class OrderValidationError(ValueError):
    """Exception raised when a proposed order violates risk or exchange rules."""


def candles_to_dataframe(candles: Sequence[Candle]) -> pd.DataFrame:
    """Convert a sequence of Candle domain objects into a validated DataFrame.

    Args:
        candles: Sequence of historical or streaming Candle records.

    Returns:
        DataFrame sorted chronologically with standardized OHLCV columns.

    Raises:
        MarketDataValidationError: If the sequence is empty.
    """
    if not candles:
        raise MarketDataValidationError("Candle sequence cannot be empty.")

    records = [
        {
            "symbol": candle.symbol,
            "open_time": candle.open_time,
            "close_time": candle.close_time,
            "open_price": float(candle.open_price),
            "high_price": float(candle.high_price),
            "low_price": float(candle.low_price),
            "close_price": float(candle.close_price),
            "volume": float(candle.volume),
            "is_closed": bool(candle.is_closed),
        }
        for candle in candles
    ]
    frame = pd.DataFrame.from_records(records)
    return (
        frame.sort_values("open_time")
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
    )


def validate_candle_dataframe(frame: pd.DataFrame, min_rows: int) -> None:
    """Validate schema, row count, price integrity, and ordering of market data.

    Args:
        frame: OHLCV DataFrame to validate.
        min_rows: Minimum number of rows required for indicator calculation.

    Raises:
        MarketDataValidationError: If schema, row count, or price invariants fail.
    """
    missing_columns = [
        col for col in REQUIRED_CANDLE_COLUMNS if col not in frame.columns
    ]
    if missing_columns:
        raise MarketDataValidationError(
            f"Missing required candle columns: {missing_columns}"
        )

    if len(frame) < min_rows:
        raise MarketDataValidationError(
            f"Insufficient candle count: got {len(frame)}, "
            f"required at least {min_rows}."
        )

    if frame[list(REQUIRED_CANDLE_COLUMNS)].isna().any().any():
        raise MarketDataValidationError(
            "Market data contains null or NaN values in required columns."
        )

    price_columns = ["open_price", "high_price", "low_price", "close_price"]
    if (frame[price_columns] <= 0.0).any().any():
        raise MarketDataValidationError(
            "All OHLC prices must be strictly positive numbers."
        )

    if (frame["volume"] < 0.0).any():
        raise MarketDataValidationError("Candle volume cannot be negative.")

    invalid_bounds = (
        (frame["high_price"] < frame["low_price"])
        | (frame["high_price"] < frame["open_price"])
        | (frame["high_price"] < frame["close_price"])
        | (frame["low_price"] > frame["open_price"])
        | (frame["low_price"] > frame["close_price"])
    )
    if invalid_bounds.any():
        raise MarketDataValidationError(
            "Candle high/low boundaries violate OHLC price invariants."
        )

    if not frame["open_time"].is_monotonic_increasing:
        raise MarketDataValidationError(
            "Candle timestamps must be strictly monotonic increasing."
        )


def validate_entry_order_viability(
    signal: MarketSignal,
    proposed_notional_usdt: float,
    available_cash_usdt: float,
    exchange_config: ExchangeConfig,
    risk_config: RiskConfig,
) -> None:
    """Validate that an entry order meets capital, exchange, and fee constraints.

    Args:
        signal: MarketSignal containing entry reference price and take profit target.
        proposed_notional_usdt: Quote amount in USDT to allocate to the order.
        available_cash_usdt: Current uninvested USDT balance.
        exchange_config: Exchange rules including minimum notional and fee rate.
        risk_config: Risk thresholds including minimum net profit ratio.

    Raises:
        OrderValidationError: If any exchange or risk constraint is violated.
    """
    if proposed_notional_usdt > available_cash_usdt:
        raise OrderValidationError(
            f"Proposed notional {proposed_notional_usdt:.4f} USDT exceeds "
            f"available cash {available_cash_usdt:.4f} USDT."
        )

    if proposed_notional_usdt < exchange_config.min_notional_usdt:
        raise OrderValidationError(
            f"Proposed notional {proposed_notional_usdt:.4f} USDT is below "
            f"exchange minimum notional {exchange_config.min_notional_usdt:.2f} USDT."
        )

    if signal.stop_loss_price >= signal.reference_price:
        raise OrderValidationError(
            "Stop loss price must be strictly below reference entry price."
        )

    if signal.take_profit_price <= signal.reference_price:
        raise OrderValidationError(
            "Take profit price must be strictly above reference entry price."
        )

    gross_return_ratio = (
        signal.take_profit_price - signal.reference_price
    ) / signal.reference_price
    round_trip_fee_ratio = 2.0 * exchange_config.taker_fee_rate
    expected_net_return_ratio = gross_return_ratio - round_trip_fee_ratio

    if expected_net_return_ratio < risk_config.min_net_profit_ratio:
        raise OrderValidationError(
            f"Expected net return ratio {expected_net_return_ratio:.5f} is below "
            f"required threshold {risk_config.min_net_profit_ratio:.5f} after fees."
        )
