from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from crypto_paper_trader.config import StrategyConfig, TradingSystemConfig
from crypto_paper_trader.domain import (
    Candle,
    MarketSignal,
    OrderSide,
    TradeExecution,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.strategy import (
    build_market_signal_from_row,
    enrich_with_indicators,
)
from crypto_paper_trader.validation import (
    candles_to_dataframe,
    validate_candle_dataframe,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestPerformanceReport:
    """Comprehensive financial and statistical summary of a historical backtest."""

    symbol: str
    interval: str
    candles_evaluated: int
    period_start: datetime
    period_end: datetime
    initial_capital_cop: float
    final_equity_cop: float
    initial_capital_usdt: float
    final_equity_usdt: float
    net_pnl_usdt: float
    net_pnl_cop: float
    net_return_pct: float
    total_closed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    total_fees_paid_usdt: float
    total_fees_paid_cop: float
    executions: tuple[TradeExecution, ...]


def compute_max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    """Calculate the maximum peak-to-trough percentage drawdown of an equity series."""
    if not equity_curve:
        return 0.0

    equity_array = np.asarray(equity_curve, dtype=float)
    running_peaks = np.maximum.accumulate(equity_array)
    drawdowns = (running_peaks - equity_array) / np.where(
        running_peaks > 0.0, running_peaks, 1.0
    )
    return round(float(np.max(drawdowns) * 100.0), 4)


def generate_vectorized_signals(
    candles: Sequence[Candle],
    symbol: str,
    config: StrategyConfig,
) -> list[tuple[Candle, MarketSignal]]:
    """Compute indicators and trading signals across a historical candle series."""
    warmup_rows = (
        max(
            config.slow_ema_period,
            config.rsi_period,
            config.atr_period,
        )
        + 2
    )
    frame = candles_to_dataframe(candles)
    validate_candle_dataframe(frame, min_rows=warmup_rows)
    enriched = enrich_with_indicators(frame, config)

    ordered_candles = sorted(candles, key=lambda item: item.open_time)
    dedup_candles: list[Candle] = []
    seen_timestamps: set[datetime] = set()
    for candle in ordered_candles:
        if candle.open_time not in seen_timestamps:
            seen_timestamps.add(candle.open_time)
            dedup_candles.append(candle)

    paired_signals: list[tuple[Candle, MarketSignal]] = []
    for idx in range(warmup_rows - 1, len(enriched)):
        row = enriched.iloc[idx]
        candle = dedup_candles[idx]
        signal = build_market_signal_from_row(row, symbol, config)
        paired_signals.append((candle, signal))

    return paired_signals


def _execute_simulation_loop(
    paired_signals: Sequence[tuple[Candle, MarketSignal]],
    engine: PaperTradingEngine,
) -> list[float]:
    """Run bar-by-bar order execution and protective exit checks over signals."""
    equity_curve: list[float] = []
    for candle, signal in paired_signals:
        active_pos = engine.active_position
        if active_pos is not None:
            if candle.low_price <= active_pos.stop_loss_price:
                engine.evaluate_protective_exits(
                    current_price=active_pos.stop_loss_price,
                    timestamp=candle.close_time,
                )
            elif candle.high_price >= active_pos.take_profit_price:
                engine.evaluate_protective_exits(
                    current_price=active_pos.take_profit_price,
                    timestamp=candle.close_time,
                )
            else:
                engine.process_signal(signal)
        else:
            engine.process_signal(signal)

        snapshot = engine.capture_portfolio_snapshot(
            current_price=candle.close_price,
            timestamp=candle.close_time,
        )
        equity_curve.append(snapshot.total_equity_usdt)

    return equity_curve


def run_historical_backtest(
    candles: Sequence[Candle],
    config: TradingSystemConfig,
    ledger_db_path: Path | None = None,
) -> BacktestPerformanceReport:
    """Execute a deterministic historical backtest using the paper trading engine."""
    paired_signals = generate_vectorized_signals(
        candles=candles,
        symbol=config.exchange.symbol,
        config=config.strategy,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        target_db = (
            ledger_db_path
            if ledger_db_path is not None
            else Path(temp_dir) / "backtest_ledger.db"
        )
        ledger = LedgerRepository(db_path=target_db)
        engine = PaperTradingEngine(
            capital_config=config.capital,
            exchange_config=config.exchange,
            risk_config=config.risk,
            ledger=ledger,
        )
        equity_curve = _execute_simulation_loop(paired_signals, engine)
        final_candle = paired_signals[-1][0]
        final_snapshot = engine.capture_portfolio_snapshot(
            current_price=final_candle.close_price,
            timestamp=final_candle.close_time,
        )
        executions = ledger.fetch_all_executions()

    sell_executions = [item for item in executions if item.side == OrderSide.SELL]
    winning_pnls = [
        item.realized_pnl_usdt
        for item in sell_executions
        if item.realized_pnl_usdt > 0.0
    ]
    losing_pnls = [
        item.realized_pnl_usdt
        for item in sell_executions
        if item.realized_pnl_usdt <= 0.0
    ]

    total_closed = len(sell_executions)
    winning_count = len(winning_pnls)
    losing_count = len(losing_pnls)
    win_rate_pct = (
        round((winning_count / total_closed) * 100.0, 2) if total_closed > 0 else 0.0
    )

    gross_profit = sum(winning_pnls)
    gross_loss = abs(sum(losing_pnls))
    if gross_loss > 0.0:
        profit_factor = round(gross_profit / gross_loss, 3)
    elif gross_profit > 0.0:
        profit_factor = 99.99
    else:
        profit_factor = 0.0

    initial_usdt = config.capital.initial_capital_usdt
    final_usdt = final_snapshot.total_equity_usdt
    net_pnl_usdt = round(final_usdt - initial_usdt, 4)
    net_pnl_cop = round(net_pnl_usdt * config.capital.cop_per_usdt_rate, 2)
    net_return_pct = round((net_pnl_usdt / initial_usdt) * 100.0, 3)
    total_fees_cop = round(
        final_snapshot.total_fees_paid_usdt * config.capital.cop_per_usdt_rate,
        2,
    )

    return BacktestPerformanceReport(
        symbol=config.exchange.symbol,
        interval=config.exchange.interval,
        candles_evaluated=len(candles),
        period_start=candles[0].open_time,
        period_end=candles[-1].close_time,
        initial_capital_cop=config.capital.initial_capital_cop,
        final_equity_cop=final_snapshot.total_equity_cop,
        initial_capital_usdt=initial_usdt,
        final_equity_usdt=final_usdt,
        net_pnl_usdt=net_pnl_usdt,
        net_pnl_cop=net_pnl_cop,
        net_return_pct=net_return_pct,
        total_closed_trades=total_closed,
        winning_trades=winning_count,
        losing_trades=losing_count,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        max_drawdown_pct=compute_max_drawdown_pct(equity_curve),
        total_fees_paid_usdt=final_snapshot.total_fees_paid_usdt,
        total_fees_paid_cop=total_fees_cop,
        executions=tuple(executions),
    )
