from __future__ import annotations

import logging
from dataclasses import dataclass

from crypto_paper_trader.config import TradingSystemConfig
from crypto_paper_trader.domain import Candle, MarketSignal, PortfolioSnapshot
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.ingestion import MarketDataProvider
from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
from crypto_paper_trader.notifications import NotificationService
from crypto_paper_trader.strategy import evaluate_market_signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleEvaluationResult:
    """Result of a single real-time market evaluation and execution cycle."""

    latest_candle: Candle
    signal: MarketSignal
    snapshot: PortfolioSnapshot


class RealTimeTradingOrchestrator:
    """Asynchronous coordinator for ingestion, strategy evaluation, and execution."""

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

    def _upsert_candle(self, candle: Candle) -> list[Candle]:
        """Insert or update a candle in the rolling window by open_time epoch."""
        epoch_key = int(candle.open_time.timestamp())
        self._candle_buffer[epoch_key] = candle

        sorted_keys = sorted(self._candle_buffer.keys())
        max_capacity = self._config.exchange.bootstrap_candle_limit + 10
        if len(sorted_keys) > max_capacity:
            for old_key in sorted_keys[:-max_capacity]:
                del self._candle_buffer[old_key]

        return [self._candle_buffer[key] for key in sorted(self._candle_buffer.keys())]

    async def bootstrap_market_history(self) -> list[Candle]:
        """Load initial historical candles into the rolling memory buffer."""
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
        """Process a single candle tick through risk checks, strategy, and execution."""
        rolling_candles = self._upsert_candle(candle)

        self._engine.evaluate_protective_exits(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )

        signal = evaluate_market_signal(
            candles=rolling_candles,
            symbol=self._config.exchange.symbol,
            config=self._config.strategy,
        )

        self._engine.process_signal(signal)

        snapshot = self._engine.capture_portfolio_snapshot(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )

        logger.info(
            "Tick processed | symbol=%s price=%.4f signal=%s rsi=%.2f "
            "fast_ema=%.4f slow_ema=%.4f atr=%.4f equity=%.4f USDT (%.0f COP)",
            signal.symbol,
            signal.reference_price,
            signal.signal_type.value,
            signal.rsi_value,
            signal.fast_ema,
            signal.slow_ema,
            signal.atr_value,
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
        """Run the end-to-end real-time paper trading loop over WebSockets."""
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
    """Assemble a RealTimeTradingOrchestrator with its ledger and engine."""
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
