from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from typing import Sequence

from crypto_paper_trader.config import TradingSystemConfig
from crypto_paper_trader.domain import (
    Candle,
    ExitReason,
    MacroMarketRegime,
    MarketSignal,
    OrderSide,
    PortfolioSnapshot,
    SignalType,
    SlotRole,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.formatting import format_price
from crypto_paper_trader.ingestion import MarketDataProvider
from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
from crypto_paper_trader.market_intelligence import (
    BtcLockState,
    MarketIntelligenceReport,
)
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
        multi_symbol_mode: bool = False,
    ) -> None:
        self._config = config
        self._provider = market_data_provider
        self._engine = trading_engine
        self._multi_symbol_mode = multi_symbol_mode
        self._candle_buffer: dict[int, Candle] = {}
        self._buffers_by_symbol: dict[str, dict[int, Candle]] = {}
        self._last_price_by_symbol: dict[str, float] = {}
        self._last_log_time: float = 0.0
        self._last_logged_signal: SignalType | None = None
        self._macro_regime: MacroMarketRegime = MacroMarketRegime.NEUTRAL
        self._cooldown_until_ts: float = 0.0
        self._cooldown_until_by_symbol: dict[str, float] = {}
        self._last_portfolio_buy_ts: float = 0.0
        self._qualified_symbols: set[str] = set()
        self._swing_candidate_symbols: set[str] = set()
        self._buy_blocked_symbols: set[str] = set()
        self._btc_lock_reasons: tuple[str, ...] = ()

    @staticmethod
    def _evaluate_anti_trap_24h_filter(
        rolling_candles: Sequence[Candle],
        max_range_position_ratio: float,
    ) -> tuple[bool, str]:
        """Reject overextended 24h-top buys and falling-knife entries below 1h EMA21.

        Args:
            rolling_candles: Chronological buffered 15m candles for the symbol.
            max_range_position_ratio: Maximum allowed percentile in the 24h range.

        Returns:
            Tuple of (passed, diagnostic_reason).
        """
        if len(rolling_candles) < 60:
            return True, ""

        window = rolling_candles[-96:]
        high_24h = max(c.high_price for c in window)
        low_24h = min(c.low_price for c in window)
        range_24h = high_24h - low_24h
        last_close = window[-1].close_price
        if range_24h <= 1e-12:
            return True, ""

        pos_in_24h = (last_close - low_24h) / range_24h
        if pos_in_24h > max_range_position_ratio:
            return (
                False,
                f"price near 24h high (range_pos={pos_in_24h:.2f} > "
                f"{max_range_position_ratio:.2f})",
            )

        alpha = 2.0 / (84.0 + 1.0)
        ema_84 = rolling_candles[0].close_price
        for item in rolling_candles[1:]:
            ema_84 = (alpha * item.close_price) + ((1.0 - alpha) * ema_84)

        if last_close < (ema_84 * 0.985) and pos_in_24h > 0.22:
            return (
                False,
                f"falling structure below 1h EMA21 (close={format_price(last_close)} "
                f"< ema84={format_price(ema_84)}, range_pos={pos_in_24h:.2f})",
            )
        return True, ""

    def set_macro_regime(self, regime: MacroMarketRegime) -> None:
        """Update the active macro market regime classification.

        Args:
            regime: MacroMarketRegime indicating overall market bias.
        """
        self._macro_regime = regime

    def set_qualified_symbols(self, symbols: set[str]) -> None:
        """Restrict multi-symbol BUY entries to top-ranked universe symbols.

        Args:
            symbols: Set of uppercase symbol identifiers approved by the scanner.
        """
        self._qualified_symbols = {
            sym.upper()
            for sym in symbols
            if sym.upper() not in self._buy_blocked_symbols
        }

    def set_swing_candidate_symbols(self, symbols: set[str]) -> None:
        """Configure universe symbols eligible to open the high-target SWING slot.

        Args:
            symbols: Set of uppercase symbol identifiers approved for Swing holds.
        """
        self._swing_candidate_symbols = {
            sym.upper()
            for sym in symbols
            if sym.upper() not in self._buy_blocked_symbols
        }
        self._engine.set_swing_candidate_symbols(self._swing_candidate_symbols)

    def set_market_intelligence(self, report: MarketIntelligenceReport) -> None:
        """Apply 3-day stabilization locks and market intelligence state to engine.

        Args:
            report: Latest MarketIntelligenceReport from MarketIntelligenceService.
        """
        blocked: set[str] = set()
        if (
            self._config.risk.enable_universal_3d_stabilization_lock
            and report.locked_symbols
        ):
            blocked.update(sym.upper() for sym in report.locked_symbols)
        if report.btc_lock_state == BtcLockState.LOCKED:
            blocked.add("BTCUSDT")
            self._btc_lock_reasons = report.reasons
        else:
            self._btc_lock_reasons = ()
        self._buy_blocked_symbols = blocked
        self._engine.set_buy_blocked_symbols(self._buy_blocked_symbols)
        if self._qualified_symbols:
            self._qualified_symbols = {
                sym
                for sym in self._qualified_symbols
                if sym not in self._buy_blocked_symbols
            }
        if self._swing_candidate_symbols:
            self._swing_candidate_symbols = {
                sym
                for sym in self._swing_candidate_symbols
                if sym not in self._buy_blocked_symbols
            }
            self._engine.set_swing_candidate_symbols(self._swing_candidate_symbols)

    def _upsert_candle(self, candle: Candle) -> list[Candle]:
        """Insert or update a candle in the rolling window by open_time epoch.

        Args:
            candle: Historical or live Candle record.

        Returns:
            Sorted list of buffered candles trimmed to bootstrap_candle_limit.
        """
        epoch_key = int(candle.open_time.timestamp())
        max_capacity = self._config.exchange.bootstrap_candle_limit + 10

        if self._multi_symbol_mode:
            sym_key = candle.symbol.upper()
            self._last_price_by_symbol[sym_key] = candle.close_price
            sym_buf = self._buffers_by_symbol.setdefault(sym_key, {})
            sym_buf[epoch_key] = candle
            sorted_keys = sorted(sym_buf.keys())
            if len(sorted_keys) > max_capacity:
                for old_key in sorted_keys[:-max_capacity]:
                    del sym_buf[old_key]
            return [sym_buf[key] for key in sorted(sym_buf.keys())]

        self._candle_buffer[epoch_key] = candle
        sorted_keys = sorted(self._candle_buffer.keys())
        if len(sorted_keys) > max_capacity:
            for old_key in sorted_keys[:-max_capacity]:
                del self._candle_buffer[old_key]

        return [self._candle_buffer[key] for key in sorted(self._candle_buffer.keys())]

    def get_buffered_universe_candles(
        self,
        min_candles: int = 60,
    ) -> dict[str, list[Candle]]:
        """Return in-memory rolling candle windows for symbols with enough history.

        Args:
            min_candles: Minimum number of buffered candles required per symbol.

        Returns:
            Mapping from uppercase symbol to its chronological list of Candles.
        """
        buffered: dict[str, list[Candle]] = {}
        for sym_key, sym_buf in self._buffers_by_symbol.items():
            if len(sym_buf) >= min_candles:
                buffered[sym_key] = [sym_buf[k] for k in sorted(sym_buf.keys())]
        return buffered

    async def bootstrap_market_history(self) -> list[Candle]:
        """Load initial historical candles into the rolling memory buffer.

        Returns:
            Chronological list of bootstrapped candles.
        """
        if self._multi_symbol_mode and any(
            len(buf) >= 30 for buf in self._buffers_by_symbol.values()
        ):
            primary = self._buffers_by_symbol.get(
                self._config.exchange.symbol.upper(), {}
            )
            return [primary[key] for key in sorted(primary.keys())]

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
        target_symbol = (
            candle.symbol.upper()
            if self._multi_symbol_mode
            else self._config.exchange.symbol
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

        if target_symbol == "BTCUSDT" and len(rolling_candles) >= min_required_rows:
            self._macro_regime = evaluate_macro_regime(
                rolling_candles, self._config.strategy
            )

        has_pos_for_symbol = self._engine.has_position(target_symbol)
        can_enter_symbol = self._engine.can_open_or_dca_symbol(
            target_symbol, candle.close_price
        )
        if (
            self._multi_symbol_mode
            and self._engine.has_any_position()
            and not has_pos_for_symbol
            and not can_enter_symbol
        ):
            snapshot = self._engine.capture_portfolio_snapshot(
                current_price=candle.close_price,
                timestamp=candle.close_time,
                symbol=target_symbol,
                prices_by_symbol=self._last_price_by_symbol,
            )
            idle_signal = MarketSignal(
                symbol=target_symbol,
                evaluated_at=candle.close_time,
                signal_type=SignalType.HOLD,
                reference_price=candle.close_price,
                stop_loss_price=candle.close_price * 0.995,
                take_profit_price=candle.close_price * 1.008,
                atr_value=candle.close_price * 0.005,
                rsi_value=50.0,
                fast_ema=candle.close_price,
                slow_ema=candle.close_price,
            )
            return CycleEvaluationResult(
                latest_candle=candle,
                signal=idle_signal,
                snapshot=snapshot,
            )

        protective_exit = self._engine.evaluate_protective_exits(
            current_price=candle.close_price,
            timestamp=candle.close_time,
            symbol=target_symbol,
        )
        if protective_exit is not None:
            cooldown_secs = 0.0
            if protective_exit.reason == ExitReason.STOP_LOSS.value:
                cooldown_secs = 900.0
            elif protective_exit.reason in (
                ExitReason.TRAILING_STOP.value,
                ExitReason.TAKE_PROFIT.value,
                ExitReason.PARTIAL_TAKE_PROFIT.value,
            ):
                cooldown_secs = self._config.risk.fast_exit_cooldown_seconds
            if cooldown_secs > 0.0:
                deadline = candle.close_time.timestamp() + cooldown_secs
                self._cooldown_until_by_symbol[target_symbol] = deadline
                if not self._multi_symbol_mode:
                    self._cooldown_until_ts = deadline

        if len(rolling_candles) < min_required_rows:
            signal = MarketSignal(
                symbol=target_symbol,
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
                symbol=target_symbol,
                config=self._config.strategy,
            )

        can_enter_now = self._engine.can_open_or_dca_symbol(
            target_symbol, candle.close_price
        )
        if (
            signal.signal_type == SignalType.BUY
            and self._engine.has_position(target_symbol)
            and not can_enter_now
        ):
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and target_symbol in self._buy_blocked_symbols
        ):
            if target_symbol == "BTCUSDT":
                reason_msg = (
                    "; ".join(self._btc_lock_reasons)
                    if self._btc_lock_reasons
                    else "mercado bajista"
                )
                logger.info(
                    "Suppressed BUY signal on %s: BTC stabilization lock active (%s).",
                    signal.symbol,
                    reason_msg,
                )
            else:
                logger.info(
                    "Suppressed BUY signal on %s: 3-day stabilization lock active "
                    "(waiting for 3d higher low + close above daily EMA10).",
                    signal.symbol,
                )
            signal = replace(signal, signal_type=SignalType.HOLD)

        active_cooldown_deadline = (
            self._cooldown_until_by_symbol.get(target_symbol, 0.0)
            if self._multi_symbol_mode
            else max(
                self._cooldown_until_ts,
                self._cooldown_until_by_symbol.get(target_symbol, 0.0),
            )
        )
        if (
            signal.signal_type == SignalType.BUY
            and candle.close_time.timestamp() < active_cooldown_deadline
        ):
            logger.info(
                "Suppressed BUY signal on %s during post-exit cooldown.",
                signal.symbol,
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._macro_regime == MacroMarketRegime.BEARISH
        ):
            is_oversold_dip = (
                signal.rsi_value <= 32.0 and signal.fast_ema >= signal.slow_ema
            )
            is_qualified_decoupled_rotation = (
                self._config.risk.no_loss_spot_mode
                and self._config.risk.allow_secondary_symbol_slot
                and target_symbol != "BTCUSDT"
                and (
                    not self._qualified_symbols
                    or target_symbol in self._qualified_symbols
                )
                and candle.close_price > candle.open_price
                and signal.fast_ema >= (signal.slow_ema * 0.9950)
                and 28.0 <= signal.rsi_value <= 54.5
            )
            if is_oversold_dip or is_qualified_decoupled_rotation:
                logger.info(
                    "Permitted %s BUY on %s during BEARISH BTC regime "
                    "(RSI=%.1f fast_ema=%s slow_ema=%s).",
                    "qualified decoupled rotation"
                    if is_qualified_decoupled_rotation
                    else "stabilized oversold dip",
                    signal.symbol,
                    signal.rsi_value,
                    format_price(signal.fast_ema),
                    format_price(signal.slow_ema),
                )
            else:
                logger.info(
                    "Suppressed BUY signal on %s: "
                    "Bitcoin macro market regime is BEARISH.",
                    signal.symbol,
                )
                signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._multi_symbol_mode
            and self._qualified_symbols
            and target_symbol not in self._qualified_symbols
        ):
            logger.info(
                "Suppressed BUY signal on %s: symbol not in top qualified "
                "universe ranking %s.",
                signal.symbol,
                sorted(self._qualified_symbols),
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._config.risk.no_loss_spot_mode
            and signal.rsi_value > 54.5
        ):
            logger.info(
                "Suppressed BUY signal on %s in no-loss spot mode: "
                "RSI=%.2f exceeds pullback ceiling 54.5.",
                signal.symbol,
                signal.rsi_value,
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._config.risk.no_loss_spot_mode
            and len(rolling_candles) >= 3
            and signal.fast_ema < signal.slow_ema
            and candle.low_price < rolling_candles[-2].low_price
            and candle.close_price <= rolling_candles[-2].high_price
            and signal.rsi_value > 33.0
        ):
            logger.info(
                "Suppressed BUY signal on %s in no-loss spot mode: "
                "unconfirmed lower-low structure while fast_ema < slow_ema.",
                signal.symbol,
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        if (
            signal.signal_type == SignalType.BUY
            and self._config.risk.enable_anti_trap_24h_filter
        ):
            passed_anti_trap, trap_reason = self._evaluate_anti_trap_24h_filter(
                rolling_candles=rolling_candles,
                max_range_position_ratio=(
                    self._config.risk.max_24h_range_position_ratio
                ),
            )
            if not passed_anti_trap:
                logger.info(
                    "Suppressed BUY signal on %s by 24h/1h anti-trap filter (%s).",
                    signal.symbol,
                    trap_reason,
                )
                signal = replace(signal, signal_type=SignalType.HOLD)

        stagger_secs = self._config.risk.portfolio_entry_stagger_seconds
        if (
            signal.signal_type == SignalType.BUY
            and stagger_secs > 0.0
            and self._engine.has_any_position()
            and self._last_portfolio_buy_ts > 0.0
            and signal.rsi_value > 30.0
            and (candle.close_time.timestamp() - self._last_portfolio_buy_ts)
            < stagger_secs
        ):
            logger.info(
                "Suppressed BUY signal on %s: portfolio entry stagger active (%.0fs).",
                signal.symbol,
                stagger_secs,
            )
            signal = replace(signal, signal_type=SignalType.HOLD)

        tm_win_prob: float | None = None
        tm_alloc: float | None = None
        can_enter_now = self._engine.can_open_or_dca_symbol(
            target_symbol, candle.close_price
        )
        if (
            protective_exit is None
            and can_enter_now
            and signal.signal_type == SignalType.BUY
            and self._config.risk.enable_time_machine_gate
        ):
            planned_role = (
                self._engine.resolve_next_slot_role(target_symbol) or SlotRole.SCALP
            )
            tm_report = run_time_machine_simulation(
                candles=rolling_candles,
                signal=signal,
                risk_config=self._config.risk,
                exchange_config=self._config.exchange,
                replays_count=self._config.risk.time_machine_replays,
                min_win_probability_pct=self._config.risk.time_machine_min_win_prob_pct,
                target_role=planned_role,
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
            signal_exec = self._engine.process_signal(
                signal=signal,
                win_probability_pct=tm_win_prob,
                allocation_fraction=tm_alloc,
            )
            if signal_exec is not None and signal_exec.side == OrderSide.BUY:
                self._last_portfolio_buy_ts = candle.close_time.timestamp()
            if (
                signal_exec is not None
                and signal_exec.reason
                in (
                    ExitReason.SIGNAL_REVERSAL.value,
                    ExitReason.BREAKEVEN_RECOVERY.value,
                )
            ):
                deadline = (
                    candle.close_time.timestamp()
                    + self._config.risk.fast_exit_cooldown_seconds
                )
                self._cooldown_until_by_symbol[target_symbol] = deadline
                if not self._multi_symbol_mode:
                    self._cooldown_until_ts = deadline

        snapshot = self._engine.capture_portfolio_snapshot(
            current_price=candle.close_price,
            timestamp=candle.close_time,
            symbol=target_symbol,
            prices_by_symbol=self._last_price_by_symbol,
        )

        wall_now = time.time()
        should_log = (
            (wall_now - self._last_log_time >= 45.0)
            or (signal.signal_type != self._last_logged_signal)
            or (candle.is_closed)
        )
        if should_log:
            self._last_log_time = wall_now
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
    multi_symbol_mode: bool = False,
) -> RealTimeTradingOrchestrator:
    """Assemble a RealTimeTradingOrchestrator with its ledger and engine.

    Args:
        config: Full trading system configuration.
        provider: MarketDataProvider implementation for REST and WebSocket IO.
        notifier: Optional mobile notification service.
        live_client: Optional authenticated Binance client for live orders.
        multi_symbol_mode: Enable simultaneous multi-asset WebSocket radar.

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
        multi_symbol_mode=multi_symbol_mode,
    )
