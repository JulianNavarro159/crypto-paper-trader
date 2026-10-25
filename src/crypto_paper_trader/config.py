from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CapitalConfig:
    """Initial capital parameters and local currency conversion settings.

    Args:
        initial_capital_cop: Starting capital in Colombian Pesos.
        cop_per_usdt_rate: Reference exchange rate from COP to USDT.
    """

    initial_capital_cop: float = 100_000.0
    cop_per_usdt_rate: float = 3_194.0

    @property
    def initial_capital_usdt(self) -> float:
        """Convert initial COP capital into USDT equivalent.

        Returns:
            Initial capital denominated in USDT.
        """
        return round(self.initial_capital_cop / self.cop_per_usdt_rate, 4)


@dataclass(frozen=True)
class UniverseConfig:
    """Multi-asset basket dividing core liquid coins and volatile momentum coins.

    Args:
        core_symbols: Five established high-liquidity cryptocurrency pairs.
        volatile_symbols: Five high-beta emerging or volatile pairs.
    """

    core_symbols: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
    )
    volatile_symbols: tuple[str, ...] = (
        "SUIUSDT",
        "DOGEUSDT",
        "PEPEUSDT",
        "AVAXUSDT",
        "LINKUSDT",
    )

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """Return the combined 10-asset universe tuple.

        Returns:
            Tuple containing both core and volatile trading pair symbols.
        """
        return self.core_symbols + self.volatile_symbols


@dataclass(frozen=True)
class ExchangeConfig:
    """Binance Spot exchange connectivity and market rule parameters.

    Args:
        symbol: Spot trading pair symbol in uppercase without separator.
        interval: Kline timeframe interval identifier.
        rest_base_url: Base HTTP URL for Binance Spot REST API.
        ws_base_url: Base WebSocket URL for Binance Spot market streams.
        bootstrap_candle_limit: Number of historical candles fetched at startup.
        taker_fee_rate: Fractional trading fee per order execution.
        min_notional_usdt: Minimum order value in USDT enforced by the exchange.
    """

    symbol: str = "SOLUSDT"
    interval: str = "1h"
    rest_base_url: str = "https://api.binance.com"
    ws_base_url: str = "wss://stream.binance.com:9443/ws"
    bootstrap_candle_limit: int = 120
    taker_fee_rate: float = 0.001
    min_notional_usdt: float = 5.0


@dataclass(frozen=True)
class StrategyConfig:
    """Hyperparameters for the multi-regime volume-climax and trend strategy.

    Args:
        fast_ema_period: Lookback window for the fast exponential moving average.
        slow_ema_period: Lookback window for the slow exponential moving average.
        trend_ema_period: Lookback window for the macro regime moving average.
        bollinger_period: Lookback window for Bollinger Bands calculation.
        bollinger_std: Standard deviation multiplier for Bollinger Bands.
        rsi_period: Lookback window for the Relative Strength Index.
        atr_period: Lookback window for the Average True Range.
        volume_climax_multiplier: Volume spike ratio relative to rolling average.
        rsi_entry_min: Minimum RSI threshold for trend pullback entries.
        rsi_entry_max: Maximum RSI threshold for trend pullback entries.
        rsi_oversold_threshold: RSI threshold identifying oversold capitulation dips.
        rsi_overbought_exit: RSI threshold that triggers an overbought profit exit.
        atr_stop_loss_multiplier: Multiplier applied to ATR for stop loss distance.
        atr_take_profit_multiplier: Multiplier applied to ATR for take profit distance.
    """

    fast_ema_period: int = 12
    slow_ema_period: int = 26
    trend_ema_period: int = 100
    bollinger_period: int = 20
    bollinger_std: float = 1.8
    rsi_period: int = 14
    atr_period: int = 14
    volume_climax_multiplier: float = 1.5
    rsi_entry_min: float = 48.0
    rsi_entry_max: float = 60.0
    rsi_oversold_threshold: float = 28.0
    rsi_overbought_exit: float = 64.0
    atr_stop_loss_multiplier: float = 2.2
    atr_take_profit_multiplier: float = 3.6


@dataclass(frozen=True)
class RiskConfig:
    """Risk management rules optimized for small-capital spot accounts.

    Args:
        position_allocation_fraction: Fraction of available cash allocated per trade.
        min_net_profit_ratio: Minimum required net profit ratio after round-trip fees.
        min_signal_exit_profit_ratio: Minimum gross gain required for signal exit.
    """

    position_allocation_fraction: float = 0.90
    min_net_profit_ratio: float = 0.004
    min_signal_exit_profit_ratio: float = 0.005


@dataclass(frozen=True)
class TradingSystemConfig:
    """Top-level configuration container for the paper trading pipeline.

    Args:
        capital: Capital and currency conversion settings.
        universe: Multi-asset basket of core and volatile symbols.
        exchange: Exchange connectivity and fee rules.
        strategy: Technical analysis indicator parameters.
        risk: Position sizing and fee-protection constraints.
        ledger_db_path: Filesystem path to the SQLite audit ledger.
    """

    capital: CapitalConfig = field(default_factory=CapitalConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ledger_db_path: Path = field(
        default_factory=lambda: Path("data") / "paper_trading_ledger.db"
    )
