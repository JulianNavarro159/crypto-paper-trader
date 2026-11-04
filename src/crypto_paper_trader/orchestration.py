from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from crypto_paper_trader.config import TradingSystemConfig
from crypto_paper_trader.domain import (
    Candle,
    ExitReason,
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
from crypto_paper_trader.simulation import run_time_machine_simulation
from crypto_paper_trader.strategy import evaluate_macro_regime, evaluate_market_signal

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
        self._cooldown_until_ts: float = 0.0

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
        if len(self._candle_buffer) >= 30:
            return [
                self._candle_buffer[key] for key in sorted(self._candle_buffer.keys())
            ]

        try:
            historical_candles = await self._provider.fetch_historical_candles()
        except Exception as exc:
            logger.warning(
                "Historical candle bootstrap skipped for %s (%s); "
                "proceeding directly to live WebSocket stream.",
                self._config.exchange.symbol,
                exc,
            )
            return []

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

        protective_exit = self._engine.evaluate_protective_exits(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )
        if (
            protective_exit is not None
            and protective_exit.reason == ExitReason.STOP_LOSS.value
        ):
            self._cooldown_until_ts = candle.close_time.timestamp() + 900.0

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
            if self._config.exchange.symbol == "BTCUSDT":
                self._macro_regime = evaluate_macro_regime(
                    rolling_candles, self._config.strategy
                )
            signal = evaluate_market_signal(
                candles=rolling_candles,
                symbol=self._config.exchange.symbol,
                config=self._config.strategy,
            )

        if (
            signal.signal_type == SignalType.BUY
            and candle.close_time.timestamp() < self._cooldown_until_ts
        ):
            logger.info(
                "Suppressed BUY signal on %s during post-stop-loss cooldown.",
                signal.symbol,
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._macro_regime == MacroMarketRegime.BEARISH
        ):
            strat_cfg = self._config.strategy
            is_oversold_dip = signal.rsi_value <= 35.0
            is_momentum_breakout = (
                strat_cfg.enable_decoupled_momentum
                and signal.rsi_value >= strat_cfg.decoupled_rsi_min
                and signal.fast_ema > signal.slow_ema
            )
            if is_oversold_dip:
                logger.info(
                    "Permitted oversold dip rebound BUY signal on %s "
                    "despite BEARISH Bitcoin macro regime (RSI=%.1f).",
                    signal.symbol,
                    signal.rsi_value,
                )
            elif is_momentum_breakout:
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

        tm_win_prob: float | None = None
        tm_alloc: float | None = None
        if (
            protective_exit is None
            and self._engine.active_position is None
            and signal.signal_type == SignalType.BUY
            and self._config.risk.enable_time_machine_gate
        ):
            tm_report = run_time_machine_simulation(
                candles=rolling_candles,
                signal=signal,
                risk_config=self._config.risk,
                exchange_config=self._config.exchange,
                replays_count=self._config.risk.time_machine_replays,
                min_win_probability_pct=self._config.risk.time_machine_min_win_prob_pct,
            )
            if not tm_report.approved:
                logger.info(
                    "Suppressed BUY signal on %s by Time Machine (%d replays): "
                    "win_prob=%.1f%% stop_prob=%.1f%% expected_net=%.3f%%.",
                    signal.symbol,
                    tm_report.replays_count,
                    tm_report.win_probability_pct,
                    tm_report.stop_hit_probability_pct,
                    tm_report.expected_net_return_pct,
                )
                signal = replace(signal, signal_type=SignalType.HOLD)
            else:
                tm_win_prob = tm_report.win_probability_pct
                tm_alloc = tm_report.recommended_allocation_fraction
                logger.info(
                    "Time Machine (%d replays) approved BUY on %s | "
                    "win_prob=%.1f%% target_prob=%.1f%% expected_net=+%.3f%% "
                    "alloc=%.0f%%",
                    tm_report.replays_count,
                    signal.symbol,
                    tm_report.win_probability_pct,
                    tm_report.target_hit_probability_pct,
                    tm_report.expected_net_return_pct,
                    tm_alloc * 100.0,
                )

        if protective_exit is None:
            self._engine.process_signal(
                signal=signal,
                win_probability_pct=tm_win_prob,
                allocation_fraction=tm_alloc,
            )

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
