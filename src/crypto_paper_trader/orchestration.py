from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from crypto_paper_trader.config import TradingSystemConfig
from crypto_paper_trader.domain import (
    Candle,
    MacroMarketRegime,
    MarketSignal,
    PortfolioSnapshot,
    SignalType,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.formatting import format_price
from crypto_paper_trader.ingestion import MarketDataProvider
from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
from crypto_paper_trader.notifications import NotificationService
from crypto_paper_trader.strategy import evaluate_market_signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleEvaluationResult:
    """Result of a single real-time market evaluation and execution cycle.

    Args:
        latest_candle: Most recent market candle processed.
        signal: Strategy signal computed from the rolling window.
        snapshot: Updated portfolio valuation snapshot.
    """

    latest_candle: Candle
    signal: MarketSignal
    snapshot: PortfolioSnapshot


class RealTimeTradingOrchestrator:
    """Asynchronous coordinator for ingestion, strategy evaluation, and execution.

    Args:
        config: Full trading system configuration.
        market_data_provider: Market data provider implementation.
        trading_engine: Stateful paper trading engine.
    """

    def __init__(
        self,
        config: TradingSystemConfig,
        market_data_provider: MarketDataProvider,
        trading_engine: PaperTradingEngine,
    ) -> None:
        self._config = config
        self._provider = market_data_provider
        self._engine = trading_engine
        self._candle_buffer: dict[int, Candle] = {}
        self._last_log_time: float = 0.0
        self._last_logged_signal: SignalType | None = None
        self._macro_regime: MacroMarketRegime = MacroMarketRegime.NEUTRAL

    def set_macro_regime(self, regime: MacroMarketRegime) -> None:
        """Update the active macro market regime classification.

        Args:
            regime: MacroMarketRegime indicating overall market bias.
        """
        self._macro_regime = regime

    def _upsert_candle(self, candle: Candle) -> list[Candle]:
        """Insert or update a candle in the rolling window by open_time epoch.

        Args:
            candle: Historical or live Candle record.

        Returns:
            Sorted list of buffered candles trimmed to bootstrap_candle_limit.
        """
        epoch_key = int(candle.open_time.timestamp())
        self._candle_buffer[epoch_key] = candle

        sorted_keys = sorted(self._candle_buffer.keys())
        max_capacity = self._config.exchange.bootstrap_candle_limit + 10
        if len(sorted_keys) > max_capacity:
            for old_key in sorted_keys[:-max_capacity]:
                del self._candle_buffer[old_key]

        return [self._candle_buffer[key] for key in sorted(self._candle_buffer.keys())]

    async def bootstrap_market_history(self) -> list[Candle]:
        """Load initial historical candles into the rolling memory buffer.

        Returns:
            Chronological list of bootstrapped candles.
        """
        historical_candles = await self._provider.fetch_historical_candles()
        ordered_candles: list[Candle] = []
        for candle in historical_candles:
            ordered_candles = self._upsert_candle(candle)

        logger.info(
            "Bootstrapped %d historical candles for %s (%s)",
            len(ordered_candles),
            self._config.exchange.symbol,
            self._config.exchange.interval,
        )
        return ordered_candles

    def process_candle_event(self, candle: Candle) -> CycleEvaluationResult:
        """Process a single candle tick through risk checks, strategy, and execution.

        Args:
            candle: Incoming live or historical candle.

        Returns:
            CycleEvaluationResult summarizing signal and portfolio equity.
        """
        rolling_candles = self._upsert_candle(candle)

        self._engine.evaluate_protective_exits(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )

        min_required_rows = (
            max(
                self._config.strategy.slow_ema_period,
                self._config.strategy.rsi_period,
                self._config.strategy.atr_period,
                self._config.strategy.bollinger_period,
            )
            + 2
        )
        if len(rolling_candles) < min_required_rows:
            signal = MarketSignal(
                symbol=self._config.exchange.symbol,
                evaluated_at=candle.close_time,
                signal_type=SignalType.HOLD,
                reference_price=candle.close_price,
                stop_loss_price=candle.close_price * 0.98,
                take_profit_price=candle.close_price * 1.04,
                atr_value=candle.close_price * 0.01,
                rsi_value=50.0,
                fast_ema=candle.close_price,
                slow_ema=candle.close_price,
            )
        else:
            signal = evaluate_market_signal(
                candles=rolling_candles,
                symbol=self._config.exchange.symbol,
                config=self._config.strategy,
            )

        if (
            signal.signal_type == SignalType.BUY
            and self._macro_regime == MacroMarketRegime.BEARISH
            and self._config.exchange.symbol != "BTCUSDT"
        ):
            strat_cfg = self._config.strategy
            if (
                strat_cfg.enable_decoupled_momentum
                and signal.rsi_value >= strat_cfg.decoupled_rsi_min
                and signal.fast_ema > signal.slow_ema
            ):
                logger.info(
                    "Permitted decoupled relative-strength BUY signal on %s "
                    "despite BEARISH Bitcoin macro regime (RSI=%.1f).",
                    signal.symbol,
                    signal.rsi_value,
                )
            else:
                logger.info(
                    "Suppressed BUY signal on %s: "
                    "Bitcoin macro market regime is BEARISH.",
                    signal.symbol,
                )
                signal = replace(signal, signal_type=SignalType.HOLD)

        self._engine.process_signal(signal)

        snapshot = self._engine.capture_portfolio_snapshot(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )

        now_ts = candle.close_time.timestamp()
        should_log = (
            (now_ts - self._last_log_time >= 45.0)
            or (signal.signal_type != self._last_logged_signal)
            or (candle.is_closed)
        )
        if should_log:
            self._last_log_time = now_ts
            self._last_logged_signal = signal.signal_type
            logger.info(
                "Tick processed | symbol=%s price=%s signal=%s rsi=%.2f "
                "fast_ema=%s slow_ema=%s atr=%s equity=%.4f USDT (%.0f COP)",
                signal.symbol,
                format_price(signal.reference_price),
                signal.signal_type.value,
                signal.rsi_value,
                format_price(signal.fast_ema),
                format_price(signal.slow_ema),
                format_price(signal.atr_value),
                snapshot.total_equity_usdt,
                snapshot.total_equity_cop,
            )
        return CycleEvaluationResult(
            latest_candle=candle,
            signal=signal,
            snapshot=snapshot,
        )

    async def run_realtime_session(
        self, max_ws_events: int | None = None
    ) -> list[CycleEvaluationResult]:
        """Run the end-to-end real-time paper trading loop over WebSockets.

        Args:
            max_ws_events: Optional cap on WebSocket events for bounded runs.

        Returns:
            List of CycleEvaluationResult objects produced during the session.
        """
        await self.bootstrap_market_history()
        results: list[CycleEvaluationResult] = []

        async for live_candle in self._provider.stream_realtime_candles(
            max_events=max_ws_events
        ):
            cycle_result = self.process_candle_event(live_candle)
            results.append(cycle_result)

        return results


def build_default_orchestrator(
    config: TradingSystemConfig,
    provider: MarketDataProvider,
    notifier: NotificationService | None = None,
    live_client: BinanceSpotAuthenticatedClient | None = None,
) -> RealTimeTradingOrchestrator:
    """Assemble a RealTimeTradingOrchestrator with its ledger and engine.

    Args:
        config: Full trading system configuration.
        provider: MarketDataProvider implementation for REST and WebSocket IO.
        notifier: Optional mobile notification service.
        live_client: Optional authenticated Binance client for live orders.

    Returns:
        Configured RealTimeTradingOrchestrator instance.
    """
    ledger = LedgerRepository(db_path=config.ledger_db_path)
    engine = PaperTradingEngine(
        capital_config=config.capital,
        exchange_config=config.exchange,
        risk_config=config.risk,
        ledger=ledger,
        notifier=notifier,
        live_client=live_client,
    )
    return RealTimeTradingOrchestrator(
        config=config,
        market_data_provider=provider,
        trading_engine=engine,
    )
