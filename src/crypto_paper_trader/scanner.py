from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from crypto_paper_trader.backtest import (
    BacktestPerformanceReport,
    compute_max_drawdown_pct,
    generate_vectorized_signals,
    run_historical_backtest,
)
from crypto_paper_trader.config import TradingSystemConfig
from crypto_paper_trader.domain import (
    Candle,
    MarketSignal,
    OrderSide,
    SignalType,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.ingestion import BinanceMarketDataClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetProjectionScore:
    """Quantitative ranking metrics for a single cryptocurrency in the universe.

    Args:
        symbol: Trading pair identifier.
        category: Basket classification (STABLE_CORE or HIGH_VOLATILITY).
        current_price: Latest closing price in USDT.
        atr_volatility_pct: Average True Range expressed as percentage of price.
        historical_return_pct: Net percentage return in backtest evaluation.
        win_rate_pct: Percentage of profitable closed trades.
        profit_factor: Ratio of gross profit to gross loss.
        closed_trades: Count of completed round-trip trades.
        composite_score: Combined ranking score for future trading selection.
    """

    symbol: str
    category: str
    current_price: float
    atr_volatility_pct: float
    historical_return_pct: float
    win_rate_pct: float
    profit_factor: float
    closed_trades: int
    composite_score: float


async def fetch_universe_candles(
    config: TradingSystemConfig,
    candle_count: int = 720,
) -> dict[str, list[Candle]]:
    """Concurrently fetch historical candles for all symbols in the universe.

    Args:
        config: Trading system configuration containing universe symbols.
        candle_count: Number of historical candles to retrieve per symbol.

    Returns:
        Dictionary mapping each symbol to its chronological list of Candles.
    """
    semaphore = asyncio.Semaphore(2)

    async def _fetch_single(symbol: str) -> tuple[str, list[Candle]]:
        async with semaphore:
            symbol_exchange_cfg = replace(config.exchange, symbol=symbol)
            client = BinanceMarketDataClient(config=symbol_exchange_cfg)
            try:
                candles = await client.fetch_paginated_historical_candles(
                    total_candles=candle_count
                )
            except Exception as exc:
                logger.warning("Failed to fetch klines for %s: %s", symbol, exc)
                candles = []
            await asyncio.sleep(0.25)
            return symbol, candles

    tasks = [_fetch_single(sym) for sym in config.universe.all_symbols]
    pairs = await asyncio.gather(*tasks)
    return {symbol: candles for symbol, candles in pairs if candles}


def evaluate_universe_projection(
    candles_by_symbol: dict[str, Sequence[Candle]],
    config: TradingSystemConfig,
) -> list[AssetProjectionScore]:
    """Evaluate and rank all symbols in the universe by quantitative projection.

    Args:
        candles_by_symbol: Mapping of symbol names to historical candle sequences.
        config: Full trading system configuration.

    Returns:
        List of AssetProjectionScore sorted descending by composite_score.
    """
    core_set = set(config.universe.core_symbols)
    scores: list[AssetProjectionScore] = []

    for symbol, candles in candles_by_symbol.items():
        symbol_config = replace(
            config,
            exchange=replace(config.exchange, symbol=symbol),
        )
        report = run_historical_backtest(candles=candles, config=symbol_config)
        paired_signals = generate_vectorized_signals(
            candles=candles,
            symbol=symbol,
            config=config.strategy,
        )
        latest_candle, latest_signal = paired_signals[-1]
        atr_pct = round(
            (latest_signal.atr_value / latest_candle.close_price) * 100.0, 4
        )
        capped_pf = min(report.profit_factor, 4.0)
        composite = round(
            (report.net_return_pct * 2.5)
            + (capped_pf * 12.0)
            + (report.win_rate_pct * 0.20)
            + (atr_pct * 6.0),
            3,
        )
        category = "STABLE_CORE" if symbol in core_set else "HIGH_VOLATILITY"
        scores.append(
            AssetProjectionScore(
                symbol=symbol,
                category=category,
                current_price=latest_candle.close_price,
                atr_volatility_pct=atr_pct,
                historical_return_pct=report.net_return_pct,
                win_rate_pct=report.win_rate_pct,
                profit_factor=report.profit_factor,
                closed_trades=report.total_closed_trades,
                composite_score=composite,
            )
        )

    return sorted(scores, key=lambda item: item.composite_score, reverse=True)


def run_multi_asset_rotation_backtest(
    candles_by_symbol: dict[str, Sequence[Candle]],
    config: TradingSystemConfig,
) -> BacktestPerformanceReport:
    """Simulate dynamic capital rotation across all 10 universe coins.

    Args:
        candles_by_symbol: Mapping of symbol names to historical candle sequences.
        config: Full trading system configuration.

    Returns:
        BacktestPerformanceReport summarizing combined multi-asset performance.
    """
    signals_by_symbol: dict[str, dict[datetime, tuple[Candle, MarketSignal]]] = {}
    all_timestamps: set[datetime] = set()

    rankings = evaluate_universe_projection(
        candles_by_symbol=candles_by_symbol,
        config=config,
    )
    qualified_scores = {
        item.symbol: item.composite_score
        for item in rankings[:3]
        if item.profit_factor >= 1.25
    }
    if not qualified_scores and rankings:
        qualified_scores = {rankings[0].symbol: rankings[0].composite_score}

    for symbol in qualified_scores:
        candles = candles_by_symbol[symbol]
        paired = generate_vectorized_signals(
            candles=candles,
            symbol=symbol,
            config=config.strategy,
        )
        timestamp_map: dict[datetime, tuple[Candle, MarketSignal]] = {}
        for candle, signal in paired:
            timestamp_map[candle.open_time] = (candle, signal)
            all_timestamps.add(candle.open_time)
        signals_by_symbol[symbol] = timestamp_map

    sorted_timestamps = sorted(all_timestamps)
    equity_curve: list[float] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        ledger = LedgerRepository(db_path=Path(temp_dir) / "rotation_ledger.db")
        engine = PaperTradingEngine(
            capital_config=config.capital,
            exchange_config=config.exchange,
            risk_config=config.risk,
            ledger=ledger,
        )

        for current_ts in sorted_timestamps:
            active_pos = engine.active_position
            if active_pos is not None:
                held_entry = signals_by_symbol.get(active_pos.symbol, {}).get(
                    current_ts
                )
                if held_entry is not None:
                    held_candle, held_signal = held_entry
                    if held_candle.low_price <= active_pos.stop_loss_price:
                        engine.evaluate_protective_exits(
                            current_price=active_pos.stop_loss_price,
                            timestamp=held_candle.close_time,
                        )
                    elif held_candle.high_price >= active_pos.take_profit_price:
                        engine.evaluate_protective_exits(
                            current_price=active_pos.take_profit_price,
                            timestamp=held_candle.close_time,
                        )
                    else:
                        engine.process_signal(held_signal)

                    mark_price = held_candle.close_price
                    mark_time = held_candle.close_time
                else:
                    mark_price = active_pos.entry_price
                    mark_time = current_ts
            else:
                buy_candidates: list[tuple[float, MarketSignal, Candle]] = []
                for symbol, proj_score in qualified_scores.items():
                    entry = signals_by_symbol.get(symbol, {}).get(current_ts)
                    if entry is None:
                        continue
                    candle, signal = entry
                    if signal.signal_type == SignalType.BUY:
                        buy_candidates.append((proj_score, signal, candle))

                mark_price = 0.0
                mark_time = current_ts
                if buy_candidates:
                    buy_candidates.sort(key=lambda item: item[0], reverse=True)
                    _, best_signal, best_candle = buy_candidates[0]
                    engine.process_signal(best_signal)
                    mark_price = best_candle.close_price
                    mark_time = best_candle.close_time

            snapshot = engine.capture_portfolio_snapshot(
                current_price=mark_price,
                timestamp=mark_time,
            )
            equity_curve.append(snapshot.total_equity_usdt)

        executions = ledger.fetch_all_executions()
        final_snapshot = snapshot
        ledger.close()

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
        symbol="UNIVERSE-10",
        interval=config.exchange.interval,
        candles_evaluated=len(sorted_timestamps),
        period_start=sorted_timestamps[0],
        period_end=sorted_timestamps[-1],
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
