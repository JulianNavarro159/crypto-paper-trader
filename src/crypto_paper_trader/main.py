from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx
from dotenv import load_dotenv

from crypto_paper_trader.backtest import (
    BacktestPerformanceReport,
    run_historical_backtest,
)
from crypto_paper_trader.config import (
    CapitalConfig,
    ExchangeConfig,
    TradingSystemConfig,
)
from crypto_paper_trader.domain import Position
from crypto_paper_trader.execution import LedgerRepository
from crypto_paper_trader.ingestion import BinanceMarketDataClient
from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
from crypto_paper_trader.notifications import build_notifier_from_env
from crypto_paper_trader.orchestration import build_default_orchestrator
from crypto_paper_trader.scanner import (
    AssetProjectionScore,
    evaluate_universe_projection,
    fetch_universe_candles,
    run_multi_asset_rotation_backtest,
)

logger = logging.getLogger(__name__)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure application-wide structured logging format.

    Args:
        level: Standard Python logging level constant.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def parse_cli_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the trading, scanning, and backtest runner.

    Args:
        argv: Optional list of CLI argument strings.

    Returns:
        Parsed Namespace containing runtime flags.
    """
    parser = argparse.ArgumentParser(
        description="Binance Spot Multi-Asset Scanner, Backtester, and Live Trader"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["realtime", "live", "verify-live", "backtest", "scan", "rotation"],
        default="realtime",
        help="Execution mode: realtime, live, verify-live, backtest, scan, or rotation",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SOLUSDT",
        help="Binance Spot trading pair symbol (default: SOLUSDT)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically scan the 10-coin universe and trade the #1 ranked coin",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Kline interval timeframe (default: 1h)",
    )
    parser.add_argument(
        "--capital-cop",
        type=float,
        default=100_000.0,
        help="Initial capital in Colombian Pesos (default: 100000.0)",
    )
    parser.add_argument(
        "--cop-rate",
        type=float,
        default=3_194.0,
        help="Reference COP per USDT exchange rate (default: 3194.0)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum WebSocket ticks to process before exiting (default: unlimited)",
    )
    parser.add_argument(
        "--backtest-candles",
        type=int,
        default=720,
        help="Historical candles to evaluate in backtest/scan mode (default: 720)",
    )
    parser.add_argument(
        "--rescan-hours",
        type=float,
        default=1.0,
        help="Hours between universe rescans in auto-select mode (default: 1.0)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data") / "paper_trading_ledger.db",
        help="Path to the SQLite audit database",
    )
    return parser.parse_args(argv)


def log_backtest_summary(report: BacktestPerformanceReport) -> None:
    """Emit structured log summary of historical backtest metrics.

    Args:
        report: Completed BacktestPerformanceReport instance.
    """
    logger.info(
        "BACKTEST SUMMARY | symbol=%s interval=%s candles=%d period=%s..%s",
        report.symbol,
        report.interval,
        report.candles_evaluated,
        report.period_start.isoformat(),
        report.period_end.isoformat(),
    )
    logger.info(
        "BACKTEST EQUITY | initial=%.0f COP (%.4f USDT) final=%.0f COP (%.4f USDT) "
        "net_pnl=%.0f COP (%.4f USDT) return=%.3f%%",
        report.initial_capital_cop,
        report.initial_capital_usdt,
        report.final_equity_cop,
        report.final_equity_usdt,
        report.net_pnl_cop,
        report.net_pnl_usdt,
        report.net_return_pct,
    )
    logger.info(
        "BACKTEST STATS | closed_trades=%d wins=%d losses=%d win_rate=%.2f%% "
        "profit_factor=%.3f max_drawdown=%.2f%% total_fees=%.0f COP (%.4f USDT)",
        report.total_closed_trades,
        report.winning_trades,
        report.losing_trades,
        report.win_rate_pct,
        report.profit_factor,
        report.max_drawdown_pct,
        report.total_fees_paid_cop,
        report.total_fees_paid_usdt,
    )


def log_universe_ranking(rankings: Sequence[AssetProjectionScore]) -> None:
    """Emit structured log table ranking the 10-coin universe by projection.

    Args:
        rankings: Sorted sequence of AssetProjectionScore records.
    """
    for rank_idx, item in enumerate(rankings, start=1):
        logger.info(
            "UNIVERSE RANK #%d | symbol=%s category=%s score=%.2f "
            "return=%+.2f%% win_rate=%.1f%% pf=%.2f atr_pct=%.2f%% trades=%d",
            rank_idx,
            item.symbol,
            item.category,
            item.composite_score,
            item.historical_return_pct,
            item.win_rate_pct,
            item.profit_factor,
            item.atr_volatility_pct,
            item.closed_trades,
        )
    if rankings:
        top = rankings[0]
        logger.info(
            "TOP PROJECTION SELECTED | symbol=%s (%s) composite_score=%.2f "
            "return=%+.2f%% profit_factor=%.2f",
            top.symbol,
            top.category,
            top.composite_score,
            top.historical_return_pct,
            top.profit_factor,
        )


async def start_cloud_health_server() -> None:
    """Start a lightweight HTTP health server if PORT is set in environment."""
    port_str = os.getenv("PORT", "").strip()
    if not port_str:
        return

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.read(1024)
            body = b"OK - CRYPTO PAPER TRADER ACTIVE\n"
            header = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain; charset=utf-8\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            )
            writer.write(header + body)
            await writer.drain()
        finally:
            writer.close()

    port = int(port_str)
    await asyncio.start_server(handle_client, host="0.0.0.0", port=port)
    logger.info("Cloud HTTP health server listening on 0.0.0.0:%d", port)

    external_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if external_url:

        async def keep_alive_ping() -> None:
            while True:
                await asyncio.sleep(540.0)
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.get(external_url)
                except httpx.HTTPError:
                    pass

        asyncio.create_task(keep_alive_ping())


async def run_application(args: argparse.Namespace) -> None:
    """Build and run the selected trading pipeline mode from CLI arguments.

    Args:
        args: Parsed command-line arguments.
    """
    load_dotenv()
    await start_cloud_health_server()
    notifier = build_notifier_from_env()

    config = TradingSystemConfig(
        capital=CapitalConfig(
            initial_capital_cop=args.capital_cop,
            cop_per_usdt_rate=args.cop_rate,
        ),
        exchange=ExchangeConfig(
            symbol=args.symbol.upper(),
            interval=args.interval,
        ),
        ledger_db_path=args.db_path,
    )

    if args.mode == "verify-live":
        live_client = BinanceSpotAuthenticatedClient.from_env(config.exchange)
        usdt_balance = live_client.fetch_available_usdt_balance()
        cop_rate = config.capital.cop_per_usdt_rate
        cash_cop = usdt_balance * cop_rate
        open_pos = LedgerRepository(db_path=config.ledger_db_path).fetch_open_position()
        coin_line = "En moneda virtual: 0.00 USDT (0 COP)"
        total_usdt = usdt_balance
        if open_pos is not None:
            base_coin = (
                open_pos.symbol[:-4]
                if open_pos.symbol.endswith("USDT")
                else open_pos.symbol
            )
            coin_qty = live_client.fetch_available_asset_balance(base_coin)
            if coin_qty <= 0.0:
                coin_qty = open_pos.quantity
            coin_usdt = coin_qty * open_pos.entry_price
            coin_cop = coin_usdt * cop_rate
            total_usdt = usdt_balance + coin_usdt
            coin_line = (
                f"En {base_coin}: {coin_qty:.4f} {base_coin} "
                f"(~{coin_usdt:.2f} USDT / {coin_cop:,.0f} COP)\n"
                f"Precio compra: {open_pos.entry_price:.4f} USDT | "
                f"Meta ganancia: {open_pos.take_profit_price:.4f} USDT"
            )
        total_cop = total_usdt * cop_rate
        logger.info(
            "LIVE BINANCE VERIFIED | cash=%.4f USDT total=%.4f USDT (~%.0f COP)",
            usdt_balance,
            total_usdt,
            total_cop,
        )
        notifier.send_alert(
            f"RESUMEN DE TU CUENTA EN BINANCE\n"
            f"En efectivo (USDT): {usdt_balance:.2f} USDT ({cash_cop:,.0f} COP)\n"
            f"{coin_line}\n"
            f"BALANCE TOTAL: {total_usdt:.2f} USDT ({total_cop:,.0f} COP)"
        )
        return

    if args.mode == "scan":
        universe_candles = await fetch_universe_candles(
            config=config,
            candle_count=args.backtest_candles,
        )
        rankings = evaluate_universe_projection(
            candles_by_symbol=universe_candles,
            config=config,
        )
        log_universe_ranking(rankings)
        return

    if args.mode == "rotation":
        universe_candles = await fetch_universe_candles(
            config=config,
            candle_count=args.backtest_candles,
        )
        rotation_report = run_multi_asset_rotation_backtest(
            candles_by_symbol=universe_candles,
            config=config,
        )
        log_backtest_summary(rotation_report)
        return

    live_client = (
        BinanceSpotAuthenticatedClient.from_env(config.exchange)
        if args.mode == "live"
        else None
    )

    existing_open_pos = (
        LedgerRepository(db_path=config.ledger_db_path).fetch_open_position()
        if args.mode in ("realtime", "live")
        else None
    )
    if existing_open_pos is None and live_client is not None:
        spot_holding = live_client.fetch_active_spot_holding(
            config.universe.all_symbols
        )
        if spot_holding is not None:
            held_sym, held_qty, held_entry = spot_holding
            notional_usdt = round(held_qty * held_entry, 4)
            existing_open_pos = Position(
                position_id=f"pos-spot-{held_sym.lower()}",
                symbol=held_sym,
                opened_at=datetime.now(tz=timezone.utc),
                entry_price=held_entry,
                quantity=held_qty,
                notional_usdt=notional_usdt,
                entry_fee_usdt=round(notional_usdt * 0.001, 6),
                stop_loss_price=round(held_entry * 0.976, 6),
                take_profit_price=round(held_entry * 1.038, 6),
            )
            LedgerRepository(db_path=config.ledger_db_path).save_open_position(
                existing_open_pos
            )

    if existing_open_pos is not None:
        config = replace(
            config,
            exchange=replace(config.exchange, symbol=existing_open_pos.symbol),
        )
    elif args.auto_select:
        try:
            universe_candles = await fetch_universe_candles(
                config=config,
                candle_count=args.backtest_candles,
            )
            rankings = evaluate_universe_projection(
                candles_by_symbol=universe_candles,
                config=config,
            )
            log_universe_ranking(rankings)
            if rankings:
                best_symbol = rankings[0].symbol
                config = replace(
                    config,
                    exchange=replace(config.exchange, symbol=best_symbol),
                )
        except Exception as exc:
            logger.warning(
                "Initial universe scan failed (%s); defaulting to %s.",
                exc,
                config.exchange.symbol,
            )

    provider = BinanceMarketDataClient(config=config.exchange)

    if args.mode == "backtest":
        historical_candles = await provider.fetch_paginated_historical_candles(
            total_candles=args.backtest_candles
        )
        report = run_historical_backtest(
            candles=historical_candles,
            config=config,
        )
        log_backtest_summary(report)
        return

    notifier.send_alert(
        f"BOT DE TRADING ACTIVO\n"
        f"Modo: {args.mode.upper()}\n"
        f"Moneda vigilada: {config.exchange.symbol} ({config.exchange.interval})"
    )

    while True:
        try:
            orchestrator = build_default_orchestrator(
                config=config,
                provider=provider,
                notifier=notifier,
                live_client=live_client,
            )
            if not args.auto_select or args.max_events is not None:
                await orchestrator.run_realtime_session(max_ws_events=args.max_events)
                return

            rescan_seconds = max(args.rescan_hours * 3600.0, 300.0)
            try:
                await asyncio.wait_for(
                    orchestrator.run_realtime_session(max_ws_events=None),
                    timeout=rescan_seconds,
                )
                return
            except TimeoutError:
                open_pos = LedgerRepository(
                    db_path=config.ledger_db_path
                ).fetch_open_position()
                if open_pos is not None:
                    logger.info(
                        "Skipping universe rescan while position in %s remains open.",
                        open_pos.symbol,
                    )
                    continue

                logger.info(
                    "Running scheduled %.1fh universe projection rescan...",
                    args.rescan_hours,
                )
                try:
                    universe_candles = await fetch_universe_candles(
                        config=config,
                        candle_count=args.backtest_candles,
                    )
                    rankings = evaluate_universe_projection(
                        candles_by_symbol=universe_candles,
                        config=config,
                    )
                    log_universe_ranking(rankings)
                    if rankings and rankings[0].symbol != config.exchange.symbol:
                        new_symbol = rankings[0].symbol
                        config = replace(
                            config,
                            exchange=replace(config.exchange, symbol=new_symbol),
                        )
                        provider = BinanceMarketDataClient(config=config.exchange)
                        notifier.send_alert(
                            f"ROTACION AUTOMATICA DE MONEDA\n"
                            f"Nueva moneda con mayor proyeccion: {new_symbol} "
                            f"({config.exchange.interval})"
                        )
                except Exception as exc:
                    logger.warning(
                        "Periodic universe rescan failed (%s); "
                        "keeping current symbol %s.",
                        exc,
                        config.exchange.symbol,
                    )
        except Exception as session_exc:
            logger.error(
                "Trading session encountered error (%s); resuming in 10s...",
                session_exc,
            )
            await asyncio.sleep(10.0)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for starting the paper trading, scanning, or live runner.

    Args:
        argv: Optional list of CLI argument strings.
    """
    configure_structured_logging()
    parsed_args = parse_cli_arguments(argv)
    try:
        asyncio.run(run_application(parsed_args))
    except KeyboardInterrupt:
        logger.info("Trading pipeline session stopped by user.")


if __name__ == "__main__":
    main()
