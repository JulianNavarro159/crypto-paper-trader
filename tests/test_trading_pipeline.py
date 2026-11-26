from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crypto_paper_trader.backtest import (
    compute_max_drawdown_pct,
    run_historical_backtest,
)
from crypto_paper_trader.config import (
    CapitalConfig,
    ExchangeConfig,
    RiskConfig,
    StrategyConfig,
    TradingSystemConfig,
)
from crypto_paper_trader.domain import (
    Candle,
    ExitReason,
    MacroMarketRegime,
    MarketSignal,
    OrderSide,
    SignalType,
    TradeExecution,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.formatting import format_price, format_quantity
from crypto_paper_trader.ingestion import (
    BinanceMarketDataClient,
    MarketDataIngestionError,
)
from crypto_paper_trader.live_exchange import quantize_step_size
from crypto_paper_trader.notifications import (
    format_balance_summary_alert,
    format_execution_alert,
    format_heartbeat_alert,
    format_recovery_alert,
    format_rotation_alert,
    format_startup_alert,
    translate_exit_reason,
)
from crypto_paper_trader.orchestration import RealTimeTradingOrchestrator
from crypto_paper_trader.scanner import (
    evaluate_universe_projection,
    run_multi_asset_rotation_backtest,
)
from crypto_paper_trader.strategy import (
    evaluate_macro_regime,
    evaluate_market_signal,
)
from crypto_paper_trader.validation import (
    MarketDataValidationError,
    OrderValidationError,
    candles_to_dataframe,
    validate_candle_dataframe,
    validate_entry_order_viability,
)


def build_synthetic_candles(
    count: int = 40,
    start_price: float = 150.0,
    step: float = 0.4,
) -> list[Candle]:
    """Generate a chronological list of synthetic OHLCV candles for testing.

    Args:
        count: Number of candles to generate.
        start_price: Initial open price.
        step: Incremental price change per candle.

    Returns:
        List of valid Candle instances.
    """
    base_time = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for idx in range(count):
        open_time = base_time + timedelta(minutes=idx)
        close_time = open_time + timedelta(seconds=59)
        open_price = round(start_price + (idx * step), 4)
        close_price = round(open_price + (step * 0.8), 4)
        high_price = round(max(open_price, close_price) + 0.5, 4)
        low_price = round(min(open_price, close_price) - 0.5, 4)
        candles.append(
            Candle(
                symbol="SOLUSDT",
                open_time=open_time,
                close_time=close_time,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=120.0 + idx,
                is_closed=True,
            )
        )
    return candles


def test_capital_conversion_from_cop_to_usdt() -> None:
    """Verify 100,000 COP converts accurately to USDT at reference rate."""
    capital = CapitalConfig(initial_capital_cop=100_000.0, cop_per_usdt_rate=4_000.0)
    assert capital.initial_capital_usdt == 25.0


def test_validate_candle_dataframe_rejects_invalid_high_low() -> None:
    """Verify OHLC validation fails when high_price is lower than low_price."""
    candles = build_synthetic_candles(count=30)
    corrupted = list(candles)
    last = corrupted[-1]
    corrupted[-1] = Candle(
        symbol=last.symbol,
        open_time=last.open_time,
        close_time=last.close_time,
        open_price=100.0,
        high_price=95.0,
        low_price=98.0,
        close_price=99.0,
        volume=10.0,
        is_closed=True,
    )
    frame = candles_to_dataframe(corrupted)
    with pytest.raises(MarketDataValidationError):
        validate_candle_dataframe(frame, min_rows=25)


def test_validate_entry_order_rejects_below_binance_min_notional() -> None:
    """Verify orders below Binance 5.0 USDT minimum notional are rejected."""
    signal = MarketSignal(
        symbol="SOLUSDT",
        evaluated_at=datetime(2026, 10, 21, 12, 30, tzinfo=timezone.utc),
        signal_type=SignalType.BUY,
        reference_price=150.0,
        stop_loss_price=147.0,
        take_profit_price=156.0,
        atr_value=1.5,
        rsi_value=55.0,
        fast_ema=149.5,
        slow_ema=148.0,
    )
    with pytest.raises(OrderValidationError):
        validate_entry_order_viability(
            signal=signal,
            proposed_notional_usdt=4.5,
            available_cash_usdt=24.0,
            exchange_config=ExchangeConfig(min_notional_usdt=5.0),
            risk_config=RiskConfig(),
        )


def test_strategy_evaluation_produces_protective_levels() -> None:
    """Verify strategy evaluation returns valid stop loss and take profit levels."""
    candles = build_synthetic_candles(count=45, start_price=140.0, step=0.25)
    signal = evaluate_market_signal(
        candles=candles,
        symbol="SOLUSDT",
        config=StrategyConfig(),
    )
    assert signal.symbol == "SOLUSDT"
    assert signal.stop_loss_price < signal.reference_price
    assert signal.take_profit_price > signal.reference_price
    assert 0.0 <= signal.rsi_value <= 100.0


def test_paper_trading_engine_executes_idempotent_buy_and_take_profit(
    tmp_path: Path,
) -> None:
    """Verify paper engine opens position, deducts fees, and realizes take profit."""
    db_path = tmp_path / "test_ledger.db"
    ledger = LedgerRepository(db_path=db_path)
    capital_config = CapitalConfig(
        initial_capital_cop=100_000.0,
        cop_per_usdt_rate=4_000.0,
    )
    exchange_config = ExchangeConfig(taker_fee_rate=0.001, min_notional_usdt=5.0)
    risk_config = RiskConfig(
        position_allocation_fraction=0.80,
        min_net_profit_ratio=0.003,
        enable_partial_take_profit=False,
    )

    engine = PaperTradingEngine(
        capital_config=capital_config,
        exchange_config=exchange_config,
        risk_config=risk_config,
        ledger=ledger,
    )

    entry_time = datetime(2026, 10, 21, 13, 0, tzinfo=timezone.utc)
    buy_signal = MarketSignal(
        symbol="SOLUSDT",
        evaluated_at=entry_time,
        signal_type=SignalType.BUY,
        reference_price=100.0,
        stop_loss_price=97.0,
        take_profit_price=106.0,
        atr_value=1.5,
        rsi_value=58.0,
        fast_ema=99.5,
        slow_ema=98.2,
    )

    buy_execution = engine.process_signal(buy_signal)
    assert buy_execution is not None
    assert buy_execution.side == OrderSide.BUY
    assert buy_execution.gross_notional_usdt == 20.0
    assert buy_execution.fee_usdt == 0.02
    assert engine.available_cash_usdt == 5.0
    assert engine.active_position is not None

    duplicate_execution = engine.process_signal(buy_signal)
    assert duplicate_execution is None

    exit_time = entry_time + timedelta(minutes=5)
    sell_execution = engine.evaluate_protective_exits(
        current_price=106.5,
        timestamp=exit_time,
    )
    assert sell_execution is not None
    assert sell_execution.side == OrderSide.SELL
    assert sell_execution.reason == ExitReason.TAKE_PROFIT.value
    assert sell_execution.realized_pnl_usdt > 1.0
    assert engine.active_position is None

    snapshot = engine.capture_portfolio_snapshot(
        current_price=106.5,
        timestamp=exit_time,
    )
    assert snapshot.total_equity_usdt > capital_config.initial_capital_usdt
    assert snapshot.total_equity_cop > capital_config.initial_capital_cop
    assert len(ledger.fetch_all_executions()) == 2


def test_compute_max_drawdown_pct_calculates_expected_decline() -> None:
    """Verify peak-to-trough percentage drawdown calculation."""
    equity_curve = [100.0, 110.0, 99.0, 105.0, 120.0]
    assert compute_max_drawdown_pct(equity_curve) == 10.0


def test_run_historical_backtest_produces_deterministic_report() -> None:
    """Verify historical backtest executes deterministically and reports metrics."""
    candles = build_synthetic_candles(count=60, start_price=100.0, step=0.15)
    config = TradingSystemConfig(
        capital=CapitalConfig(
            initial_capital_cop=100_000.0,
            cop_per_usdt_rate=4_000.0,
        )
    )
    report = run_historical_backtest(candles=candles, config=config)
    assert report.candles_evaluated == 60
    assert report.initial_capital_cop == 100_000.0
    assert report.initial_capital_usdt == 25.0
    assert report.final_equity_usdt > 0.0
    assert report.max_drawdown_pct >= 0.0


def test_universe_projection_and_multi_asset_rotation() -> None:
    """Verify 10-coin universe ranking and multi-asset rotation backtest."""
    candles_sol = build_synthetic_candles(count=60, start_price=100.0, step=0.20)
    candles_btc = build_synthetic_candles(count=60, start_price=80000.0, step=10.0)
    universe_map = {"SOLUSDT": candles_sol, "BTCUSDT": candles_btc}
    config = TradingSystemConfig(
        capital=CapitalConfig(
            initial_capital_cop=100_000.0,
            cop_per_usdt_rate=4_000.0,
        )
    )
    rankings = evaluate_universe_projection(
        candles_by_symbol=universe_map,
        config=config,
    )
    assert len(rankings) == 2
    assert rankings[0].composite_score >= rankings[1].composite_score

    rotation_report = run_multi_asset_rotation_backtest(
        candles_by_symbol=universe_map,
        config=config,
    )
    assert rotation_report.symbol == "UNIVERSE-10"
    assert rotation_report.initial_capital_usdt == 25.0


def test_quantize_step_size_and_mobile_alert_formatting() -> None:
    """Verify Binance LOT_SIZE step quantization and mobile alert formatting."""
    assert quantize_step_size(18.625441, "0.10000000") == "18.60000000"
    assert quantize_step_size(0.0002659, "0.00001000") == "0.00026000"

    candles = build_synthetic_candles(count=60, start_price=100.0, step=0.20)
    config = TradingSystemConfig()
    report = run_historical_backtest(candles=candles, config=config)
    if report.executions:
        alert_text = format_execution_alert(
            execution=report.executions[0],
            available_cash_usdt=24.0,
            cop_per_usdt_rate=4_150.0,
            is_live_order=True,
        )
        assert "BINANCE REAL" in alert_text
        assert "BALANCE TOTAL" in alert_text


def test_trailing_stop_locks_profit_on_reversal() -> None:
    """Verify trailing stop ratchets stop loss to protect profit on reversal."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ledger = LedgerRepository(db_path=Path(temp_dir) / "test_ts.db")
        try:
            engine = PaperTradingEngine(
                capital_config=CapitalConfig(
                    initial_capital_cop=100_000.0, cop_per_usdt_rate=4_000.0
                ),
                exchange_config=ExchangeConfig(symbol="DOGEUSDT"),
                risk_config=RiskConfig(),
                ledger=ledger,
            )
            now = datetime.now(tz=timezone.utc)
            buy_signal = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=now,
                signal_type=SignalType.BUY,
                reference_price=100.0,
                stop_loss_price=96.0,
                take_profit_price=108.0,
                atr_value=1.5,
                rsi_value=52.0,
                fast_ema=100.0,
                slow_ema=98.0,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None

            engine.evaluate_protective_exits(current_price=103.0, timestamp=now)
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price >= 101.5
            assert pos.peak_price == 103.0

            sell_exec = engine.evaluate_protective_exits(
                current_price=101.5, timestamp=now
            )
            assert sell_exec is not None
            assert sell_exec.reason == ExitReason.TRAILING_STOP.value
            assert sell_exec.realized_pnl_usdt > 0.0
            assert engine.active_position is None
        finally:
            ledger.close()


def test_macro_market_regime_classification_and_orchestrator_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify macro Bitcoin regime gates altcoin BUY signals during downtrends."""
    config = StrategyConfig()
    bullish_candles = build_synthetic_candles(
        count=45, start_price=60_000.0, step=150.0
    )
    bullish_regime = evaluate_macro_regime(bullish_candles, config)
    assert bullish_regime == MacroMarketRegime.BULLISH

    bearish_candles = build_synthetic_candles(
        count=45, start_price=70_000.0, step=-200.0
    )
    bearish_regime = evaluate_macro_regime(bearish_candles, config)
    assert bearish_regime == MacroMarketRegime.BEARISH

    with tempfile.TemporaryDirectory() as temp_dir:
        ledger = LedgerRepository(db_path=Path(temp_dir) / "test.db")
        try:
            trading_engine = PaperTradingEngine(
                capital_config=CapitalConfig(),
                exchange_config=ExchangeConfig(symbol="SOLUSDT"),
                risk_config=RiskConfig(),
                ledger=ledger,
            )
            trading_system_config = TradingSystemConfig(
                capital=CapitalConfig(),
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=RiskConfig(),
                strategy=config,
            )

            class DummyProvider:
                pass

            orchestrator = RealTimeTradingOrchestrator(
                config=trading_system_config,
                market_data_provider=DummyProvider(),
                trading_engine=trading_engine,
            )

            now = datetime.now(tz=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=now,
                signal_type=SignalType.BUY,
                reference_price=150.0,
                stop_loss_price=145.0,
                take_profit_price=160.0,
                atr_value=2.0,
                rsi_value=55.0,
                fast_ema=150.0,
                slow_ema=148.0,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: buy_signal,
            )

            for candle in bullish_candles:
                orchestrator._upsert_candle(candle)

            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)
            bearish_result = orchestrator.process_candle_event(bullish_candles[-1])
            assert bearish_result.signal.signal_type == SignalType.HOLD
            assert trading_engine.active_position is None

            orchestrator.set_macro_regime(MacroMarketRegime.BULLISH)
            bullish_result = orchestrator.process_candle_event(bullish_candles[-1])
            assert bullish_result.signal.signal_type == SignalType.BUY
            assert trading_engine.active_position is not None
        finally:
            ledger.close()


def test_ntfy_retry_on_rate_limit_and_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NtfyNotificationService retries on HTTP 429 and delivers alert."""
    import httpx

    from crypto_paper_trader.notifications import NtfyNotificationService

    calls = 0

    def mock_post(self: object, url: str, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            req = httpx.Request("POST", url)
            return httpx.Response(
                status_code=429,
                headers={"Retry-After": "0"},
                request=req,
            )
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, request=req)

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    service = NtfyNotificationService(topic="test_topic")
    success = service.send_alert("Test Alert")
    assert success is True
    assert calls == 2


def test_adaptive_price_and_quantity_formatting() -> None:
    """Validate dynamic formatting for crypto prices and trade quantities."""
    assert format_price(0.00000945) == "0.00000945"
    assert format_price(0.184521) == "0.184521"
    assert format_price(2.3456) == "2.3456"
    assert format_price(749.78) == "749.7800"
    assert format_price(94500.5) == "94500.50"
    assert format_price(0.0) == "0.00"

    assert format_quantity(3200000.0) == "3,200,000"
    assert format_quantity(3200000.45) == "3,200,000.45"
    assert format_quantity(0.0357) == "0.035700"
    assert format_quantity(12.5) == "12.5000"


def test_format_execution_alert_with_micro_prices() -> None:
    """Ensure trade alerts for micro-value coins avoid truncating to zero."""
    from crypto_paper_trader.domain import OrderSide, TradeExecution

    execution = TradeExecution(
        client_order_id="test-buy-pepe",
        position_id="pos-pepe-1",
        symbol="PEPEUSDT",
        side=OrderSide.BUY,
        executed_at=datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc),
        execution_price=0.00000950,
        quantity=3000000.0,
        gross_notional_usdt=28.50,
        fee_usdt=0.0285,
        realized_pnl_usdt=0.0,
        reason="SIGNAL_ENTRY",
    )
    alert = format_execution_alert(
        execution=execution,
        available_cash_usdt=4.36,
        cop_per_usdt_rate=3194.0,
        is_live_order=True,
        stop_loss_price=0.00000925,
        take_profit_price=0.00001010,
    )
    assert "0.00000950 USDT" in alert
    assert "3,000,000 PEPE" in alert
    assert "0.00000925 USDT" in alert
    assert "0.00001010 USDT" in alert
    assert "0.0000 USDT" not in alert


def test_partial_take_profit_and_adaptive_trailing() -> None:
    """Validate two-stage take-profit execution and breakeven adjustment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_ledger.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            capital_cfg = CapitalConfig(
                initial_capital_cop=100000.0,
                cop_per_usdt_rate=3194.0,
            )
            exchange_cfg = ExchangeConfig(symbol="LINKUSDT", taker_fee_rate=0.001)
            risk_cfg = RiskConfig(
                enable_partial_take_profit=True,
                tp1_profit_ratio=0.048,
                tp1_close_fraction=0.50,
                tp2_take_profit_ratio=0.090,
            )
            engine = PaperTradingEngine(
                capital_config=capital_cfg,
                exchange_config=exchange_cfg,
                risk_config=risk_cfg,
                ledger=ledger,
            )

            entry_time = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="LINKUSDT",
                evaluated_at=entry_time,
                signal_type=SignalType.BUY,
                reference_price=10.0,
                stop_loss_price=9.70,
                take_profit_price=10.48,
                atr_value=0.20,
                rsi_value=55.0,
                fast_ema=10.1,
                slow_ema=9.9,
            )
            buy_execution = engine.process_signal(buy_signal)
            assert buy_execution is not None
            assert engine.active_position is not None
            initial_qty = engine.active_position.quantity

            tp1_time = datetime(2026, 10, 30, 11, 0, tzinfo=timezone.utc)
            tp1_execution = engine.evaluate_protective_exits(
                current_price=10.50,
                timestamp=tp1_time,
            )
            assert tp1_execution is not None
            assert tp1_execution.reason == ExitReason.PARTIAL_TAKE_PROFIT.value
            assert tp1_execution.realized_pnl_usdt > 0.0

            pos = engine.active_position
            assert pos is not None
            assert pos.is_partially_closed is True
            assert pos.quantity < initial_qty
            assert pos.stop_loss_price >= 10.004

            tp2_time = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
            final_execution = engine.evaluate_protective_exits(
                current_price=10.95,
                timestamp=tp2_time,
            )
            assert final_execution is not None
            assert final_execution.reason == ExitReason.TAKE_PROFIT.value
            assert engine.active_position is None
        finally:
            ledger.close()


def test_decoupled_momentum_under_bearish_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify overbought momentum signals are suppressed during bearish macro regime."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_ledger.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            candles = build_synthetic_candles(count=35, start_price=10.0, step=0.1)
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="LINKUSDT"),
                ledger_db_path=db_path,
            )
            trading_engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(config=config.exchange)
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=trading_engine,
            )
            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)

            for candle in candles[:-1]:
                orchestrator._upsert_candle(candle)

            decoupled_signal = MarketSignal(
                symbol="LINKUSDT",
                evaluated_at=candles[-1].close_time,
                signal_type=SignalType.BUY,
                reference_price=11.0,
                stop_loss_price=10.6,
                take_profit_price=11.6,
                atr_value=0.25,
                rsi_value=61.0,
                fast_ema=11.2,
                slow_ema=10.8,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: decoupled_signal,
            )

            result = orchestrator.process_candle_event(candles[-1])
            assert result.signal.signal_type == SignalType.HOLD
            assert trading_engine.active_position is None
        finally:
            ledger.close()


def test_friendly_spanish_recovery_alert_reassures_user() -> None:
    """Verify error recovery alerts omit raw HTTP codes and reassure in Spanish."""
    binance_err = RuntimeError(
        "All Binance market data clusters failed: Rate limited: HTTP 418"
    )
    alert_418 = format_recovery_alert(error=binance_err, sleep_seconds=120.0)
    assert "PAUSA PREVENTIVA POR ALTO TRAFICO EN BINANCE" in alert_418
    assert "100% seguro" in alert_418
    assert "HTTP 418" not in alert_418
    assert "RuntimeError" not in alert_418

    net_err = ConnectionError("Connection refused by peer")
    alert_net = format_recovery_alert(error=net_err, sleep_seconds=30.0)
    assert "PAUSA PREVENTIVA POR INTERMITENCIA DE RED" in alert_net
    assert "100% seguro" in alert_net


def test_friendly_spanish_heartbeat_and_startup_alerts() -> None:
    """Ensure periodic health and startup messages are fully clear in Spanish."""
    heartbeat_active = format_heartbeat_alert(
        symbol="LINKUSDT",
        interval="1h",
        open_position_symbol="LINKUSDT",
        open_position_entry_price=10.5220,
        available_cash_usdt=3.21,
        cop_per_usdt_rate=3194.0,
        total_realized_pnl_usdt=1.25,
        closed_trades_count=2,
        win_rate_pct=100.0,
    )
    assert "REPORTE PERIODICO DE ESTADO 24/7" in heartbeat_active
    assert "Comprado en LINK a 10.5220 USDT" in heartbeat_active
    assert "3.21 USDT" in heartbeat_active
    assert "10,253 COP" in heartbeat_active

    startup_alert = format_startup_alert(
        mode="live",
        symbol="SOLUSDT",
        interval="1h",
    )
    assert "BOT DE TRADING INICIADO CON EXITO" in startup_alert
    assert "Operaciones reales con dinero real en Binance" in startup_alert
    assert "SOL" in startup_alert


def test_friendly_spanish_execution_alerts() -> None:
    """Verify trade executions produce clear, jargon-free explanations in Spanish."""
    exec_record = TradeExecution(
        client_order_id="test-sell-link",
        position_id="pos-link-1",
        symbol="LINKUSDT",
        side=OrderSide.SELL,
        executed_at=datetime(2026, 10, 30, 13, 0, tzinfo=timezone.utc),
        execution_price=11.3000,
        quantity=2.6700,
        gross_notional_usdt=30.1710,
        fee_usdt=0.0302,
        realized_pnl_usdt=2.0470,
        reason="PARTIAL_TAKE_PROFIT",
    )
    partial_alert = format_execution_alert(
        execution=exec_record,
        available_cash_usdt=17.50,
        cop_per_usdt_rate=3194.0,
        is_live_order=True,
    )
    assert "PRIMERA META ALCANZADA: GANANCIA ASEGURADA" in partial_alert
    assert "Vendio la mitad (50%) de tus LINK" in partial_alert
    assert "+2.0470 USDT" in partial_alert


def test_friendly_spanish_balance_summary_and_rotation_alerts() -> None:
    """Verify account balance summaries and currency rotations format cleanly."""
    summary = format_balance_summary_alert(
        available_usdt=25.0,
        cop_per_usdt_rate=3200.0,
        open_position_coin="LINK",
        open_position_qty=2.67,
        open_position_usdt=28.10,
        open_entry_price=10.522,
        open_take_profit=11.300,
    )
    assert "RESUMEN DE TU CUENTA EN BINANCE" in summary
    assert "BALANCE TOTAL: 53.10 USDT" in summary

    rotation = format_rotation_alert(
        old_symbol="SOLUSDT",
        new_symbol="PEPEUSDT",
        interval="1h",
    )
    assert "CAMBIO AUTOMATICO DE MONEDA VIGILADA" in rotation
    assert "PEPE" in rotation

    reason = translate_exit_reason("TAKE_PROFIT")
    assert "Meta de ganancia" in reason


def test_micro_candle_15m_scalping_and_profit_lock() -> None:
    """Verify 15m scalping locks in profit above 100 COP when gain reaches 0.35%."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ledger = LedgerRepository(db_path=Path(temp_dir) / "test_micro.db")
        try:
            capital_cfg = CapitalConfig(
                initial_capital_cop=112_000.0,
                cop_per_usdt_rate=3_200.0,
            )
            exchange_cfg = ExchangeConfig(
                symbol="LINKUSDT",
                interval="15m",
                taker_fee_rate=0.001,
            )
            risk_cfg = RiskConfig(
                tp1_profit_ratio=0.0075,
                tp1_close_fraction=0.50,
                tp2_take_profit_ratio=0.0180,
                min_net_profit_ratio=0.0015,
            )
            engine = PaperTradingEngine(
                capital_config=capital_cfg,
                exchange_config=exchange_cfg,
                risk_config=risk_cfg,
                ledger=ledger,
            )

            entry_time = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="LINKUSDT",
                evaluated_at=entry_time,
                signal_type=SignalType.BUY,
                reference_price=10.0000,
                stop_loss_price=9.8700,
                take_profit_price=10.1800,
                atr_value=0.0800,
                rsi_value=34.0,
                fast_ema=10.0100,
                slow_ema=9.9900,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None

            tick_time = datetime(2026, 11, 1, 12, 5, tzinfo=timezone.utc)
            engine.evaluate_protective_exits(current_price=10.0400, timestamp=tick_time)
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price >= 10.0200

            tp1_time = datetime(2026, 11, 1, 12, 10, tzinfo=timezone.utc)
            tp1_exec = engine.evaluate_protective_exits(
                current_price=10.0800, timestamp=tp1_time
            )
            assert tp1_exec is not None
            assert tp1_exec.reason == ExitReason.PARTIAL_TAKE_PROFIT.value
            net_cop = tp1_exec.realized_pnl_usdt * 3_200.0
            assert net_cop > 100.0
        finally:
            ledger.close()


def test_orchestrator_handles_bootstrap_failure_gracefully() -> None:
    """Verify orchestrator recovers gracefully when REST bootstrap encounters 418."""

    async def _runner() -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_ledger.db"
            ledger = LedgerRepository(db_path=db_path)
            try:
                config = TradingSystemConfig(
                    exchange=ExchangeConfig(symbol="SOLUSDT"),
                    ledger_db_path=db_path,
                )
                trading_engine = PaperTradingEngine(
                    capital_config=config.capital,
                    exchange_config=config.exchange,
                    risk_config=config.risk,
                    ledger=ledger,
                )

                class FailingBootstrapProvider:
                    async def fetch_historical_candles(self) -> list[Candle]:
                        raise MarketDataIngestionError("Simulated 418 IP ban on REST")

                    async def stream_realtime_candles(
                        self, max_events: int | None = None
                    ):
                        candle = build_synthetic_candles(count=1)[0]
                        yield candle

                orchestrator = RealTimeTradingOrchestrator(
                    config=config,
                    market_data_provider=FailingBootstrapProvider(),
                    trading_engine=trading_engine,
                )
                results = await orchestrator.run_realtime_session(max_ws_events=1)
                assert len(results) == 1
                assert results[0].latest_candle.symbol == "SOLUSDT"
                assert results[0].signal.signal_type == SignalType.HOLD
            finally:
                ledger.close()

    asyncio.run(_runner())


def test_oversold_dip_rebound_permitted_under_bearish_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify oversold dip buys require trend stabilization in bearish regime."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_ledger.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            candles = build_synthetic_candles(count=35, start_price=10.0, step=-0.1)
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SUIUSDT"),
                ledger_db_path=db_path,
            )
            trading_engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(config=config.exchange)
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=trading_engine,
            )
            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)

            for candle in candles[:-1]:
                orchestrator._upsert_candle(candle)

            falling_knife_signal = MarketSignal(
                symbol="SUIUSDT",
                evaluated_at=candles[-1].close_time,
                signal_type=SignalType.BUY,
                reference_price=2.0000,
                stop_loss_price=1.9800,
                take_profit_price=2.0400,
                atr_value=0.0300,
                rsi_value=32.0,
                fast_ema=1.9900,
                slow_ema=2.0200,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: falling_knife_signal,
            )

            result = orchestrator.process_candle_event(candles[-1])
            assert result.signal.signal_type == SignalType.HOLD
            assert trading_engine.active_position is None

            stabilized_signal = MarketSignal(
                symbol="SUIUSDT",
                evaluated_at=candles[-1].close_time,
                signal_type=SignalType.BUY,
                reference_price=2.0000,
                stop_loss_price=1.9800,
                take_profit_price=2.0400,
                atr_value=0.0300,
                rsi_value=31.5,
                fast_ema=2.0250,
                slow_ema=2.0200,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: stabilized_signal,
            )

            stabilized_result = orchestrator.process_candle_event(candles[-1])
            assert stabilized_result.signal.signal_type == SignalType.BUY
            assert trading_engine.active_position is not None
        finally:
            ledger.close()


def test_signal_generation_enforces_minimum_fee_protective_spread() -> None:
    """Verify market signal generation ensures take profit covers round-trip fees."""
    import pandas as pd

    from crypto_paper_trader.strategy import build_market_signal_from_row

    row_data = {
        "close_price": 100.0,
        "atr": 0.0001,
        "rsi": 35.0,
        "fast_ema": 99.8,
        "slow_ema": 100.2,
        "signal_type": "BUY",
        "close_time": datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc),
    }
    series_row = pd.Series(row_data)
    cfg = StrategyConfig()
    signal = build_market_signal_from_row(series_row, "SOLUSDT", cfg)
    gross_return = (
        signal.take_profit_price - signal.reference_price
    ) / signal.reference_price
    assert gross_return >= 0.0074
    sl_ratio = (
        signal.reference_price - signal.stop_loss_price
    ) / signal.reference_price
    assert 0.0039 <= sl_ratio <= 0.0051

    risk_cfg = RiskConfig()
    exchange_cfg = ExchangeConfig()
    validate_entry_order_viability(
        signal=signal,
        proposed_notional_usdt=25.0,
        available_cash_usdt=30.0,
        exchange_config=exchange_cfg,
        risk_config=risk_cfg,
    )


def test_btc_micro_profit_lock_and_post_stop_loss_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify micro-profit lock at +0.22% and signal reversal exit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_btc_lock.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="BTCUSDT"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            base_time = datetime(2026, 11, 3, 20, 39, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="BTCUSDT",
                evaluated_at=base_time,
                signal_type=SignalType.BUY,
                reference_price=77890.24,
                stop_loss_price=77422.90,
                take_profit_price=78474.42,
                atr_value=160.0,
                rsi_value=58.0,
                fast_ema=77850.0,
                slow_ema=77700.0,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None

            engine.evaluate_protective_exits(
                current_price=78070.00,
                timestamp=base_time + timedelta(minutes=65),
            )
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price >= 78007.0

            sell_signal = MarketSignal(
                symbol="BTCUSDT",
                evaluated_at=base_time + timedelta(minutes=66),
                signal_type=SignalType.SELL,
                reference_price=78140.00,
                stop_loss_price=77500.0,
                take_profit_price=78600.0,
                atr_value=150.0,
                rsi_value=69.64,
                fast_ema=77890.0,
                slow_ema=77742.0,
            )
            exit_exec = engine.process_signal(sell_signal)
            assert exit_exec is not None
            assert exit_exec.reason == ExitReason.SIGNAL_REVERSAL.value
            assert engine.active_position is None
        finally:
            ledger.close()


def test_bnb_live_order_preserves_full_quantity_and_accurate_pnl() -> None:
    """Verify BNBUSDT live buy preserves full executed quantity and positive PnL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_bnb_live.db"
        ledger = LedgerRepository(db_path=db_path)
        try:

            class StubLiveClient:
                def __init__(self) -> None:
                    self._known_usdt: float | None = 35.18
                    self._assets: dict[str, float] = {"BNB": 0.0005}

                def get_known_usdt_balance(self) -> float | None:
                    return self._known_usdt

                def update_known_usdt_balance(self, usdt_balance: float) -> None:
                    self._known_usdt = usdt_balance

                def get_known_asset_balance(self, asset: str) -> float:
                    return self._assets.get(asset.upper(), 0.0)

                def adjust_known_asset_balance(self, asset: str, delta: float) -> None:
                    key = asset.upper()
                    self._assets[key] = max(0.0, self._assets.get(key, 0.0) + delta)

                def is_cooldown_active(self) -> bool:
                    return False

                def fetch_symbol_step_size(self, symbol: str) -> str:
                    return "0.00100000"

                def place_market_buy_by_quote(
                    self, symbol: str, quote_amount_usdt: float, client_order_id: str
                ) -> dict[str, object]:
                    return {
                        "executedQty": "0.04000000",
                        "cummulativeQuoteQty": "27.95800000",
                        "fills": [
                            {
                                "price": "698.95000000",
                                "qty": "0.04000000",
                                "commission": "0.00003000",
                                "commissionAsset": "BNB",
                            }
                        ],
                    }

                def place_market_sell_by_quantity(
                    self, symbol: str, quantity: float, client_order_id: str
                ) -> dict[str, object]:
                    return {
                        "executedQty": f"{quantity:.8f}",
                        "cummulativeQuoteQty": f"{quantity * 701.20:.8f}",
                        "fills": [],
                    }

            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="BNBUSDT"),
                ledger_db_path=db_path,
            )
            stub_client = StubLiveClient()
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
                live_client=stub_client,
            )
            assert engine.available_cash_usdt == 35.18

            base_time = datetime(2026, 11, 4, 1, 10, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="BNBUSDT",
                evaluated_at=base_time,
                signal_type=SignalType.BUY,
                reference_price=698.95,
                stop_loss_price=694.75,
                take_profit_price=704.19,
                atr_value=1.50,
                rsi_value=56.0,
                fast_ema=698.90,
                slow_ema=698.50,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None
            assert engine.active_position.quantity == 0.040

            sell_signal = MarketSignal(
                symbol="BNBUSDT",
                evaluated_at=base_time + timedelta(minutes=25),
                signal_type=SignalType.SELL,
                reference_price=701.20,
                stop_loss_price=695.0,
                take_profit_price=705.0,
                atr_value=1.50,
                rsi_value=62.5,
                fast_ema=700.10,
                slow_ema=699.50,
            )
            sell_exec = engine.process_signal(sell_signal)
            assert sell_exec is not None
            assert sell_exec.quantity == 0.040
            assert sell_exec.realized_pnl_usdt > 0.0
        finally:
            ledger.close()


def test_time_machine_10000_replays_approves_high_edge_and_blocks_negative_edge() -> (
    None
):
    """Verify 10,000-replay Time Machine approves bullish edge and blocks bad setups."""
    from crypto_paper_trader.simulation import run_time_machine_simulation

    candles = build_synthetic_candles(count=60, start_price=100.0, step=0.20)
    risk_cfg = RiskConfig()
    exchange_cfg = ExchangeConfig(symbol="SOLUSDT")
    now = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)

    strong_signal = MarketSignal(
        symbol="SOLUSDT",
        evaluated_at=now,
        signal_type=SignalType.BUY,
        reference_price=112.0,
        stop_loss_price=111.1,
        take_profit_price=113.2,
        atr_value=0.6,
        rsi_value=54.0,
        fast_ema=112.1,
        slow_ema=111.4,
    )
    approved_report = run_time_machine_simulation(
        candles=candles,
        signal=strong_signal,
        risk_config=risk_cfg,
        exchange_config=exchange_cfg,
        replays_count=10_000,
    )
    assert approved_report.replays_count == 10_000
    assert approved_report.approved is True
    assert approved_report.win_probability_pct >= 55.0
    assert approved_report.expected_net_return_pct > 0.0
    assert approved_report.recommended_allocation_fraction >= 0.70

    weak_signal = MarketSignal(
        symbol="SOLUSDT",
        evaluated_at=now,
        signal_type=SignalType.BUY,
        reference_price=112.0,
        stop_loss_price=111.8,
        take_profit_price=114.5,
        atr_value=0.8,
        rsi_value=69.5,
        fast_ema=111.0,
        slow_ema=112.2,
    )
    rejected_report = run_time_machine_simulation(
        candles=candles,
        signal=weak_signal,
        risk_config=risk_cfg,
        exchange_config=exchange_cfg,
        replays_count=10_000,
    )
    assert rejected_report.approved is False
    assert rejected_report.recommended_allocation_fraction == 0.0

    exec_record = TradeExecution(
        client_order_id="test-tm-alert",
        position_id="pos-tm-1",
        symbol="SOLUSDT",
        side=OrderSide.BUY,
        executed_at=now,
        execution_price=112.0,
        quantity=0.25,
        gross_notional_usdt=28.0,
        fee_usdt=0.028,
        realized_pnl_usdt=0.0,
        reason="SIGNAL_ENTRY",
    )
    alert_text = format_execution_alert(
        execution=exec_record,
        available_cash_usdt=7.28,
        cop_per_usdt_rate=3200.0,
        is_live_order=True,
        stop_loss_price=111.1,
        take_profit_price=113.2,
        win_probability_pct=approved_report.win_probability_pct,
    )
    assert "Simulacion Maquina del Tiempo (10,000 escenarios):" in alert_text


def test_opening_range_breakout_entry_timer_triggers_on_box_breakout() -> None:
    """Verify Opening-Range Breakout Entry Timer detects compressed box breakouts."""
    base_time = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for idx in range(30):
        open_t = base_time + timedelta(minutes=idx * 15)
        close_t = open_t + timedelta(minutes=14, seconds=59)
        offset = 0.04 if idx % 2 == 0 else -0.03
        price = 100.0 + offset
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_t,
                close_time=close_t,
                open_price=round(price - 0.01, 4),
                high_price=round(price + 0.08, 4),
                low_price=round(price - 0.08, 4),
                close_price=round(price, 4),
                volume=100.0,
                is_closed=True,
            )
        )
    breakout_open_t = base_time + timedelta(minutes=30 * 15)
    breakout_close_t = breakout_open_t + timedelta(minutes=14, seconds=59)
    candles.append(
        Candle(
            symbol="BTCUSDT",
            open_time=breakout_open_t,
            close_time=breakout_close_t,
            open_price=100.02,
            high_price=100.19,
            low_price=100.00,
            close_price=100.16,
            volume=135.0,
            is_closed=True,
        )
    )
    signal = evaluate_market_signal(
        candles=candles,
        symbol="BTCUSDT",
        config=StrategyConfig(),
    )
    assert signal.signal_type == SignalType.BUY


def test_overbought_entry_blocked_and_hair_trigger_net_profit_lock() -> None:
    """Verify overbought RSI=63.81 is rejected and +0.23% peak locks net profit."""
    from crypto_paper_trader.simulation import run_time_machine_simulation

    candles = build_synthetic_candles(count=60, start_price=100.0, step=0.15)
    now = datetime(2026, 11, 4, 16, 18, tzinfo=timezone.utc)
    overbought_signal = MarketSignal(
        symbol="ADAUSDT",
        evaluated_at=now,
        signal_type=SignalType.BUY,
        reference_price=0.5000,
        stop_loss_price=0.4968,
        take_profit_price=0.5042,
        atr_value=0.0030,
        rsi_value=63.81,
        fast_ema=0.5002,
        slow_ema=0.4990,
    )
    tm_report = run_time_machine_simulation(
        candles=candles,
        signal=overbought_signal,
        risk_config=RiskConfig(),
        exchange_config=ExchangeConfig(symbol="ADAUSDT"),
        replays_count=10_000,
    )
    assert tm_report.approved is False

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_hair_trigger.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="ADAUSDT", taker_fee_rate=0.001),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            valid_buy = MarketSignal(
                symbol="ADAUSDT",
                evaluated_at=now,
                signal_type=SignalType.BUY,
                reference_price=1.0000,
                stop_loss_price=0.9940,
                take_profit_price=1.0080,
                atr_value=0.0040,
                rsi_value=49.0,
                fast_ema=1.0005,
                slow_ema=0.9995,
            )
            buy_exec = engine.process_signal(valid_buy)
            assert buy_exec is not None

            engine.evaluate_protective_exits(
                current_price=1.0024,
                timestamp=now + timedelta(minutes=2),
            )
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price >= 1.0022

            exit_exec = engine.evaluate_protective_exits(
                current_price=1.0022,
                timestamp=now + timedelta(minutes=3),
            )
            assert exit_exec is not None
            assert exit_exec.reason == ExitReason.TRAILING_STOP.value
            assert exit_exec.realized_pnl_usdt > 0.0
        finally:
            ledger.close()


def test_universal_cooldown_enforced_across_all_protective_and_signal_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cooldown timers are strictly enforced after all exit reasons."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cooldown.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                ledger_db_path=db_path,
            )
            trading_engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(config=config.exchange)
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=trading_engine,
            )
            now = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)

            def mock_buy_signal(candles, symbol, config):
                latest = candles[-1]
                return MarketSignal(
                    symbol=symbol,
                    evaluated_at=latest.close_time,
                    signal_type=SignalType.BUY,
                    reference_price=latest.close_price,
                    stop_loss_price=round(latest.close_price * 0.995, 4),
                    take_profit_price=round(latest.close_price * 1.015, 4),
                    atr_value=0.5,
                    rsi_value=48.0,
                    fast_ema=round(latest.close_price + 0.1, 4),
                    slow_ema=round(latest.close_price - 0.1, 4),
                )

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_buy_signal,
            )
            candles = build_synthetic_candles(count=35, start_price=100.0, step=0.0)
            for c in candles[:-1]:
                orchestrator._upsert_candle(c)

            first_result = orchestrator.process_candle_event(candles[-1])
            assert first_result.signal.signal_type == SignalType.BUY
            assert trading_engine.active_position is not None

            stop_candle = Candle(
                symbol="SOLUSDT",
                open_time=now + timedelta(minutes=15),
                close_time=now + timedelta(minutes=30),
                open_price=100.0,
                high_price=100.1,
                low_price=99.4,
                close_price=99.4,
                volume=500.0,
                is_closed=True,
            )
            stop_result = orchestrator.process_candle_event(stop_candle)
            assert stop_result.latest_candle.close_price == 99.4
            assert trading_engine.active_position is None

            suppressed_candle = Candle(
                symbol="SOLUSDT",
                open_time=now + timedelta(minutes=30),
                close_time=now + timedelta(minutes=44),
                open_price=99.8,
                high_price=100.2,
                low_price=99.7,
                close_price=100.0,
                volume=300.0,
                is_closed=True,
            )
            suppressed_result = orchestrator.process_candle_event(suppressed_candle)
            assert suppressed_result.signal.signal_type == SignalType.HOLD
            assert trading_engine.active_position is None

            permitted_candle = Candle(
                symbol="SOLUSDT",
                open_time=now + timedelta(minutes=45),
                close_time=now + timedelta(minutes=60),
                open_price=100.0,
                high_price=100.2,
                low_price=99.9,
                close_price=100.1,
                volume=350.0,
                is_closed=True,
            )
            permitted_result = orchestrator.process_candle_event(permitted_candle)
            assert permitted_result.signal.signal_type == SignalType.BUY
            assert trading_engine.active_position is not None

            reversal_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=now + timedelta(minutes=65),
                signal_type=SignalType.SELL,
                reference_price=100.41,
                stop_loss_price=99.5,
                take_profit_price=102.0,
                atr_value=0.5,
                rsi_value=62.0,
                fast_ema=100.3,
                slow_ema=100.0,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: reversal_signal,
            )
            reversal_candle = Candle(
                symbol="SOLUSDT",
                open_time=now + timedelta(minutes=60),
                close_time=now + timedelta(minutes=65),
                open_price=100.1,
                high_price=100.41,
                low_price=100.1,
                close_price=100.41,
                volume=400.0,
                is_closed=True,
            )
            reversal_result = orchestrator.process_candle_event(reversal_candle)
            assert reversal_result.signal.signal_type == SignalType.SELL
            assert trading_engine.active_position is None

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_buy_signal,
            )
            post_reversal_suppressed = Candle(
                symbol="SOLUSDT",
                open_time=now + timedelta(minutes=65),
                close_time=now + timedelta(minutes=68),
                open_price=100.4,
                high_price=100.6,
                low_price=100.3,
                close_price=100.4,
                volume=200.0,
                is_closed=True,
            )
            post_result = orchestrator.process_candle_event(post_reversal_suppressed)
            assert post_result.signal.signal_type == SignalType.HOLD
            assert trading_engine.active_position is None
        finally:
            ledger.close()


def test_signal_reversal_guarantees_strictly_positive_net_pnl_above_fees() -> None:
    """Verify signal reversal exits suppress sub-fee gains and guarantee
    positive net profit.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_reversal_fee.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(
                    symbol="SOLUSDT",
                    taker_fee_rate=0.001,
                ),
                risk=RiskConfig(
                    min_signal_exit_profit_ratio=0.0028,
                    min_net_profit_ratio=0.0010,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            now = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=now,
                signal_type=SignalType.BUY,
                reference_price=100.0,
                stop_loss_price=99.5,
                take_profit_price=102.0,
                atr_value=0.5,
                rsi_value=50.0,
                fast_ema=100.1,
                slow_ema=99.9,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None

            sub_fee_sell = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=now + timedelta(minutes=5),
                signal_type=SignalType.SELL,
                reference_price=100.18,
                stop_loss_price=99.5,
                take_profit_price=102.0,
                atr_value=0.5,
                rsi_value=59.0,
                fast_ema=100.1,
                slow_ema=99.9,
            )
            sub_fee_exec = engine.process_signal(sub_fee_sell)
            assert sub_fee_exec is None
            assert engine.active_position is not None

            fee_immune_sell = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=now + timedelta(minutes=10),
                signal_type=SignalType.SELL,
                reference_price=100.32,
                stop_loss_price=99.5,
                take_profit_price=102.0,
                atr_value=0.5,
                rsi_value=60.0,
                fast_ema=100.2,
                slow_ema=99.9,
            )
            valid_exit = engine.process_signal(fee_immune_sell)
            assert valid_exit is not None
            assert valid_exit.reason == ExitReason.SIGNAL_REVERSAL.value
            assert valid_exit.realized_pnl_usdt > 0.0
            assert engine.active_position is None
        finally:
            ledger.close()


def test_trailing_stop_tier2_positive_net_profit_lock() -> None:
    """Verify tier 2 trailing stop locks net profits strictly above round-trip fees."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tier2_lock.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(
                    symbol="DOGEUSDT",
                    taker_fee_rate=0.001,
                ),
                risk=RiskConfig(
                    min_net_profit_ratio=0.0010,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            now = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=now,
                signal_type=SignalType.BUY,
                reference_price=0.100000,
                stop_loss_price=0.099500,
                take_profit_price=0.101000,
                atr_value=0.000400,
                rsi_value=48.0,
                fast_ema=0.100050,
                slow_ema=0.099950,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None

            engine.evaluate_protective_exits(
                current_price=0.100240,
                timestamp=now + timedelta(minutes=2),
            )
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price >= 0.100220

            exit_exec = engine.evaluate_protective_exits(
                current_price=0.100220,
                timestamp=now + timedelta(minutes=4),
            )
            assert exit_exec is not None
            assert exit_exec.reason == ExitReason.TRAILING_STOP.value
            assert exit_exec.realized_pnl_usdt > 0.0
            assert engine.active_position is None
        finally:
            ledger.close()


def test_ntfy_long_retry_after_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify long ntfy 429 Retry-After headers return immediately without blocking."""
    import time as time_mod

    import httpx

    from crypto_paper_trader.notifications import NtfyNotificationService

    calls = 0

    def mock_post(self: object, url: str, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "935"},
            request=req,
        )

    monkeypatch.setattr(httpx.Client, "post", mock_post)
    service = NtfyNotificationService(topic="test_topic")
    start_ts = time_mod.monotonic()
    first_result = service.send_alert("Alert 1")
    second_result = service.send_alert("Alert 2")
    elapsed = time_mod.monotonic() - start_ts

    assert first_result is False
    assert second_result is False
    assert calls == 1
    assert elapsed < 2.0


def test_multi_symbol_radar_executes_cross_symbol_entry_and_isolates_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 10-coin multi-symbol radar enters cross-symbol BUY and isolates exits."""
    from dataclasses import replace

    from crypto_paper_trader.ingestion import parse_ws_kline_payload

    combined_payload = {
        "stream": "solusdt@kline_15m",
        "data": {
            "k": {
                "s": "SOLUSDT",
                "t": 1793900000000,
                "T": 1793900899999,
                "o": "100.00",
                "h": "100.50",
                "l": "99.90",
                "c": "100.20",
                "v": "500.0",
                "x": False,
            }
        },
    }
    parsed = parse_ws_kline_payload(combined_payload)
    assert parsed.symbol == "SOLUSDT"
    assert parsed.close_price == 100.20

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_multi_radar.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="ADAUSDT"),
                risk=RiskConfig(
                    enable_time_machine_gate=False,
                    enable_partial_take_profit=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(
                config=config.exchange,
                symbols=("ADAUSDT", "SOLUSDT", "BTCUSDT"),
            )
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )

            sol_candles = [
                replace(c, symbol="SOLUSDT")
                for c in build_synthetic_candles(count=35, start_price=100.0, step=0.1)
            ]
            for c in sol_candles[:-1]:
                orchestrator._upsert_candle(c)

            sol_buy = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=sol_candles[-1].close_time,
                signal_type=SignalType.BUY,
                reference_price=103.40,
                stop_loss_price=102.88,
                take_profit_price=104.20,
                atr_value=0.40,
                rsi_value=50.0,
                fast_ema=103.35,
                slow_ema=103.10,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: sol_buy,
            )

            res_entry = orchestrator.process_candle_event(sol_candles[-1])
            assert res_entry.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None
            assert engine.active_position.symbol == "SOLUSDT"

            ada_tick = Candle(
                symbol="ADAUSDT",
                open_time=sol_candles[-1].open_time + timedelta(minutes=1),
                close_time=sol_candles[-1].close_time + timedelta(minutes=1),
                open_price=0.35,
                high_price=0.36,
                low_price=0.34,
                close_price=0.35,
                volume=1000.0,
                is_closed=False,
            )
            res_ada = orchestrator.process_candle_event(ada_tick)
            assert res_ada.signal.signal_type == SignalType.HOLD
            assert engine.active_position is not None
            assert engine.active_position.symbol == "SOLUSDT"
            assert res_ada.snapshot.total_equity_usdt > 28.0

            sol_tp_tick = Candle(
                symbol="SOLUSDT",
                open_time=sol_candles[-1].open_time + timedelta(minutes=2),
                close_time=sol_candles[-1].close_time + timedelta(minutes=2),
                open_price=104.00,
                high_price=104.35,
                low_price=103.95,
                close_price=104.30,
                volume=800.0,
                is_closed=False,
            )
            orchestrator.process_candle_event(sol_tp_tick)
            assert engine.active_position is None
            execs = ledger.fetch_all_executions()
            assert len(execs) >= 2
            assert execs[-1].symbol == "SOLUSDT"
            assert execs[-1].realized_pnl_usdt > 0.0
        finally:
            ledger.close()


def test_spot_no_loss_micro_profit_never_sells_below_entry_and_banks_micro_gain() -> (
    None
):
    """Verify no-loss Spot mode enforces floor >= entry and sells at +0.01 USDT net."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_no_loss_spot.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            capital_cfg = CapitalConfig(
                initial_capital_cop=102_200.0,
                cop_per_usdt_rate=3_194.0,
            )
            exchange_cfg = ExchangeConfig(symbol="ETHUSDT", taker_fee_rate=0.001)
            risk_cfg = RiskConfig.spot_no_loss_micro_profit(
                min_target_net_profit_usdt=0.01
            )
            engine = PaperTradingEngine(
                capital_config=capital_cfg,
                exchange_config=exchange_cfg,
                risk_config=risk_cfg,
                ledger=ledger,
            )
            initial_cash = engine.available_cash_usdt
            t0 = datetime(2026, 11, 7, 6, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="ETHUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=2246.97,
                stop_loss_price=2237.98,
                take_profit_price=2263.82,
                atr_value=8.5,
                rsi_value=44.0,
                fast_ema=2247.10,
                slow_ema=2246.50,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price > pos.entry_price
            assert pos.take_profit_price >= pos.stop_loss_price

            buy_alert = format_execution_alert(
                execution=buy_exec,
                available_cash_usdt=engine.available_cash_usdt,
                cop_per_usdt_rate=capital_cfg.cop_per_usdt_rate,
                is_live_order=True,
                stop_loss_price=pos.stop_loss_price,
                take_profit_price=pos.take_profit_price,
            )
            assert "nunca vende menor a tu compra" in buy_alert

            for idx, dip_price in enumerate([2246.97, 2245.00, 2237.32, 2215.00]):
                dip_res = engine.evaluate_protective_exits(
                    current_price=dip_price,
                    timestamp=t0 + timedelta(minutes=idx + 1),
                )
                assert dip_res is None
                assert engine.active_position is not None

            for idx, sub_fee_sell_price in enumerate([2237.32, 2246.97, 2249.00]):
                sell_sig = MarketSignal(
                    symbol="ETHUSDT",
                    evaluated_at=t0 + timedelta(minutes=10 + idx),
                    signal_type=SignalType.SELL,
                    reference_price=sub_fee_sell_price,
                    stop_loss_price=2230.0,
                    take_profit_price=2260.0,
                    atr_value=8.5,
                    rsi_value=62.0,
                    fast_ema=2247.0,
                    slow_ema=2246.5,
                )
                assert engine.process_signal(sell_sig) is None
                assert engine.active_position is not None

            target_price = pos.take_profit_price
            win_exec = engine.evaluate_protective_exits(
                current_price=target_price,
                timestamp=t0 + timedelta(minutes=20),
            )
            assert win_exec is not None
            assert win_exec.side == OrderSide.SELL
            assert win_exec.execution_price > buy_exec.execution_price
            assert win_exec.realized_pnl_usdt >= 0.01
            assert engine.available_cash_usdt >= round(initial_cash + 0.01, 4)
            assert engine.active_position is None
        finally:
            ledger.close()


def test_spot_no_loss_upgrades_resumed_position_and_blocks_top_chasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify resumed positions upgrade floor >= entry and high-RSI buys are blocked."""
    from crypto_paper_trader.config import UniverseConfig
    from crypto_paper_trader.domain import Position

    assert "BNBUSDT" not in UniverseConfig().all_symbols
    assert "LTCUSDT" in UniverseConfig().core_symbols

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_upgrade_no_loss.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)
            legacy_pos = Position(
                position_id="pos-legacy-1",
                symbol="SOLUSDT",
                opened_at=t0,
                entry_price=100.00,
                quantity=0.28,
                notional_usdt=28.00,
                entry_fee_usdt=0.028,
                stop_loss_price=99.50,
                take_profit_price=100.80,
                peak_price=100.00,
            )
            ledger.save_open_position(legacy_pos)

            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT", taker_fee_rate=0.001),
                risk=RiskConfig.spot_no_loss_micro_profit(),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            resumed = engine.active_position
            assert resumed is not None
            assert resumed.stop_loss_price > 100.00

            engine._active_position = None
            ledger.clear_open_position()

            provider = BinanceMarketDataClient(config=config.exchange)
            orchestrator = RealTimeTradingOrchestrator(
                config=replace(
                    config,
                    risk=replace(config.risk, enable_time_machine_gate=False),
                ),
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            candles = build_synthetic_candles(count=65, start_price=100.0, step=0.1)
            for c in candles[:-1]:
                orchestrator._upsert_candle(c)

            buffered_map = orchestrator.get_buffered_universe_candles(min_candles=60)
            assert "SOLUSDT" in buffered_map
            assert len(buffered_map["SOLUSDT"]) >= 60

            high_rsi_buy = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=candles[-1].close_time,
                signal_type=SignalType.BUY,
                reference_price=106.40,
                stop_loss_price=105.88,
                take_profit_price=107.20,
                atr_value=0.40,
                rsi_value=56.0,
                fast_ema=106.35,
                slow_ema=106.10,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: high_rsi_buy,
            )
            suppressed = orchestrator.process_candle_event(candles[-1])
            assert suppressed.signal.signal_type == SignalType.HOLD
            assert engine.active_position is None
        finally:
            ledger.close()


def test_spot_no_loss_backtest_and_time_machine_record_zero_losing_sales() -> None:
    """Verify backtest and Time Machine under no-loss Spot mode record zero losses."""
    from crypto_paper_trader.simulation import run_time_machine_simulation

    candles = build_synthetic_candles(count=70, start_price=100.0, step=0.20)
    config = TradingSystemConfig(
        risk=RiskConfig.spot_no_loss_micro_profit(),
    )
    report = run_historical_backtest(candles=candles, config=config)
    assert report.losing_trades == 0
    for exec_item in report.executions:
        if exec_item.side == OrderSide.SELL:
            assert exec_item.realized_pnl_usdt >= 0.01

    signal = MarketSignal(
        symbol="SOLUSDT",
        evaluated_at=candles[-1].close_time,
        signal_type=SignalType.BUY,
        reference_price=candles[-1].close_price,
        stop_loss_price=candles[-1].close_price * 0.995,
        take_profit_price=candles[-1].close_price * 1.008,
        atr_value=0.60,
        rsi_value=45.0,
        fast_ema=candles[-1].close_price + 0.1,
        slow_ema=candles[-1].close_price - 0.1,
    )
    tm_res = run_time_machine_simulation(
        candles=candles,
        signal=signal,
        risk_config=config.risk,
        exchange_config=config.exchange,
        replays_count=5_000,
    )
    assert tm_res.replays_count == 5_000
    assert tm_res.approved is True
    assert tm_res.win_probability_pct >= 55.0
    assert tm_res.expected_net_return_pct > 0.0


def test_sniper_anti_dip_and_qualified_universe_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify universe qualification gate and anti-dip lower-low filter block traps."""
    from crypto_paper_trader.scanner import (
        AssetProjectionScore,
        select_qualified_universe_symbols,
    )
    from crypto_paper_trader.simulation import run_time_machine_simulation

    scores = [
        AssetProjectionScore(
            symbol="SOLUSDT",
            category="STABLE_CORE",
            current_price=83.22,
            atr_volatility_pct=0.45,
            historical_return_pct=0.39,
            win_rate_pct=100.0,
            profit_factor=99.99,
            closed_trades=3,
            composite_score=76.48,
        ),
        AssetProjectionScore(
            symbol="AVAXUSDT",
            category="HIGH_VOLATILITY",
            current_price=8.60,
            atr_volatility_pct=0.52,
            historical_return_pct=0.25,
            win_rate_pct=100.0,
            profit_factor=99.99,
            closed_trades=2,
            composite_score=62.10,
        ),
        AssetProjectionScore(
            symbol="ADAUSDT",
            category="HIGH_VOLATILITY",
            current_price=0.1736,
            atr_volatility_pct=0.30,
            historical_return_pct=-0.45,
            win_rate_pct=0.0,
            profit_factor=0.0,
            closed_trades=0,
            composite_score=-12.82,
        ),
    ]
    qualified = select_qualified_universe_symbols(scores, max_symbols=5)
    assert qualified == {"SOLUSDT", "AVAXUSDT"}

    falling_candles = build_synthetic_candles(
        count=40, start_price=100.0, step=-0.35
    )
    falling_signal = MarketSignal(
        symbol="ADAUSDT",
        evaluated_at=falling_candles[-1].close_time,
        signal_type=SignalType.BUY,
        reference_price=falling_candles[-1].close_price,
        stop_loss_price=falling_candles[-1].close_price * 0.995,
        take_profit_price=falling_candles[-1].close_price * 1.008,
        atr_value=0.45,
        rsi_value=42.0,
        fast_ema=falling_candles[-1].close_price - 0.1,
        slow_ema=falling_candles[-1].close_price + 0.5,
    )
    tm_falling = run_time_machine_simulation(
        candles=falling_candles,
        signal=falling_signal,
        risk_config=RiskConfig.spot_no_loss_micro_profit(),
        exchange_config=ExchangeConfig(),
        replays_count=5_000,
    )
    assert tm_falling.approved is False

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_sniper_filter.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=replace(
                    RiskConfig.spot_no_loss_micro_profit(),
                    enable_time_machine_gate=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(config=config.exchange)
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            orchestrator.set_qualified_symbols(qualified)

            ada_candles = [
                replace(c, symbol="ADAUSDT")
                for c in build_synthetic_candles(count=35, start_price=1.0, step=0.01)
            ]
            for c in ada_candles[:-1]:
                orchestrator._upsert_candle(c)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: MarketSignal(
                    symbol="ADAUSDT",
                    evaluated_at=ada_candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=ada_candles[-1].close_price,
                    stop_loss_price=ada_candles[-1].close_price * 0.995,
                    take_profit_price=ada_candles[-1].close_price * 1.008,
                    atr_value=0.01,
                    rsi_value=44.0,
                    fast_ema=ada_candles[-1].close_price,
                    slow_ema=ada_candles[-1].close_price * 0.999,
                ),
            )
            res_unqualified = orchestrator.process_candle_event(ada_candles[-1])
            assert res_unqualified.signal.signal_type == SignalType.HOLD
            assert engine.active_position is None
        finally:
            ledger.close()


def test_no_loss_trailing_runner_captures_extended_gains_above_safety_floor() -> None:
    """Verify No-Loss Trailing Runner lets strong surges run above fee floor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_trailing_runner.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT", taker_fee_rate=0.001),
                risk=RiskConfig.spot_no_loss_trailing_runner(btc_unlock_mode="NONE"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            t0 = datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=72.00,
                stop_loss_price=71.50,
                take_profit_price=72.60,
                atr_value=0.35,
                rsi_value=44.0,
                fast_ema=72.05,
                slow_ema=71.90,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            pos = engine.get_open_position("SOLUSDT")
            assert pos is not None
            initial_cash = config.capital.initial_capital_usdt
            assert 14.0 <= buy_exec.gross_notional_usdt <= 16.5
            assert engine.available_cash_usdt >= 15.0

            assert (
                engine.evaluate_protective_exits(
                    current_price=72.22,
                    timestamp=t0 + timedelta(minutes=1),
                    symbol="SOLUSDT",
                )
                is None
            )
            assert (
                engine.evaluate_protective_exits(
                    current_price=73.44,
                    timestamp=t0 + timedelta(minutes=2),
                    symbol="SOLUSDT",
                )
                is None
            )
            sell_exec = engine.evaluate_protective_exits(
                current_price=73.15,
                timestamp=t0 + timedelta(minutes=3),
                symbol="SOLUSDT",
            )
            assert sell_exec is not None
            assert sell_exec.reason == ExitReason.TRAILING_STOP.value
            assert sell_exec.realized_pnl_usdt > 0.15
            assert engine.available_cash_usdt > initial_cash
            assert engine.get_open_position("SOLUSDT") is None
        finally:
            ledger.close()


def test_dual_slot_parallel_altcoin_trading_while_btc_position_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify secondary altcoin slots trade concurrently while BTC awaits recovery."""
    from crypto_paper_trader.domain import Position

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_dual_slot.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 17, 10, 0, tzinfo=timezone.utc)
            btc_pos = Position(
                position_id="reconciled-btcusdt",
                symbol="BTCUSDT",
                opened_at=t0 - timedelta(days=5),
                entry_price=73_852.00,
                quantity=0.000190,
                notional_usdt=14.03188,
                entry_fee_usdt=0.014,
                stop_loss_price=74_060.86,
                take_profit_price=74_086.78,
                peak_price=73_852.00,
            )
            ledger.save_open_position_for_symbol(btc_pos)

            config = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=96_400.0,
                    cop_per_usdt_rate=3_194.0,
                ),
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=replace(
                    RiskConfig.spot_no_loss_trailing_runner(btc_unlock_mode="NONE"),
                    enable_time_machine_gate=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            assert engine.has_position("BTCUSDT") is True
            assert engine.available_cash_usdt > 15.5
            assert engine.can_open_position_for_symbol("SOLUSDT") is True

            provider = BinanceMarketDataClient(
                config=config.exchange,
                symbols=("BTCUSDT", "SOLUSDT"),
            )
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            sol_candles = [
                replace(c, symbol="SOLUSDT")
                for c in build_synthetic_candles(count=35, start_price=70.0, step=0.08)
            ]
            for c in sol_candles[:-1]:
                orchestrator._upsert_candle(c)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: MarketSignal(
                    symbol="SOLUSDT",
                    evaluated_at=sol_candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=sol_candles[-1].close_price,
                    stop_loss_price=sol_candles[-1].close_price * 0.995,
                    take_profit_price=sol_candles[-1].close_price * 1.008,
                    atr_value=0.35,
                    rsi_value=45.0,
                    fast_ema=sol_candles[-1].close_price,
                    slow_ema=sol_candles[-1].close_price * 0.999,
                ),
            )
            entry_res = orchestrator.process_candle_event(sol_candles[-1])
            assert entry_res.signal.signal_type == SignalType.BUY
            assert engine.has_position("BTCUSDT") is True
            assert engine.has_position("SOLUSDT") is True

            sol_surge = Candle(
                symbol="SOLUSDT",
                open_time=sol_candles[-1].open_time + timedelta(minutes=15),
                close_time=sol_candles[-1].close_time + timedelta(minutes=15),
                open_price=sol_candles[-1].close_price,
                high_price=sol_candles[-1].close_price * 1.015,
                low_price=sol_candles[-1].close_price,
                close_price=sol_candles[-1].close_price * 1.015,
                volume=500.0,
                is_closed=False,
            )
            orchestrator.process_candle_event(sol_surge)

            sol_pullback = Candle(
                symbol="SOLUSDT",
                open_time=sol_candles[-1].open_time + timedelta(minutes=16),
                close_time=sol_candles[-1].close_time + timedelta(minutes=16),
                open_price=sol_candles[-1].close_price * 1.015,
                high_price=sol_candles[-1].close_price * 1.015,
                low_price=sol_candles[-1].close_price * 1.010,
                close_price=sol_candles[-1].close_price * 1.010,
                volume=400.0,
                is_closed=False,
            )
            orchestrator.process_candle_event(sol_pullback)
            assert engine.has_position("SOLUSDT") is False
            assert engine.has_position("BTCUSDT") is True
            execs = ledger.fetch_all_executions()
            assert execs[-1].symbol == "SOLUSDT"
            assert execs[-1].realized_pnl_usdt > 0.04
        finally:
            ledger.close()


def test_partial_unlock_and_tranche_allocation_preserves_liquidity() -> None:
    """Verify idempotent 50% BTC unlock and post-partial Spot reconciliation."""
    from crypto_paper_trader.domain import Position
    from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_partial_unlock.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 12, 3, 9, 35, tzinfo=timezone.utc)
            btc_pos = Position(
                position_id="reconciled-btcusdt",
                symbol="BTCUSDT",
                opened_at=t0,
                entry_price=73_852.00,
                quantity=0.000380,
                notional_usdt=28.06376,
                entry_fee_usdt=0.02806,
                stop_loss_price=74_030.22,
                take_profit_price=74_056.13,
                peak_price=73_852.00,
            )
            ledger.save_open_position_for_symbol(btc_pos)

            config = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=102_270.0,
                    cop_per_usdt_rate=3_194.0,
                ),
                exchange=ExchangeConfig(symbol="BTCUSDT"),
                risk=RiskConfig.spot_no_loss_trailing_runner(
                    btc_unlock_mode="PARTIAL_50"
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            unlock_exec = engine.execute_controlled_partial_unlock(
                symbol="BTCUSDT",
                fraction=0.50,
                current_price=64_250.00,
                timestamp=datetime(2026, 11, 17, 16, 15, tzinfo=timezone.utc),
            )
            assert unlock_exec is not None
            assert unlock_exec.reason == "PARTIAL_UNLOCK"
            assert unlock_exec.quantity == pytest.approx(0.000190, abs=1e-8)
            assert engine.available_cash_usdt >= 16.0

            duplicate_unlock = engine.execute_controlled_partial_unlock(
                symbol="BTCUSDT",
                fraction=0.50,
                current_price=64_250.00,
            )
            assert duplicate_unlock is None

            remaining_btc = engine.get_open_position("BTCUSDT")
            assert remaining_btc is not None
            assert remaining_btc.quantity == pytest.approx(0.000190, abs=1e-8)
            assert remaining_btc.stop_loss_price > 73_852.00

            client = BinanceSpotAuthenticatedClient(
                config=ExchangeConfig(),
                api_key="k",
                api_secret="s",
            )
            client._step_size_cache["BTCUSDT"] = "0.00001000"

            def fake_request(
                method: str, path: str, params: dict[str, object]
            ) -> dict[str, object]:
                if path == "/api/v3/account":
                    return {
                        "balances": [
                            {"asset": "USDT", "free": "16.1500"},
                            {"asset": "BTC", "free": "0.00019961"},
                        ]
                    }
                if path == "/api/v3/myTrades" and params.get("symbol") == "BTCUSDT":
                    return {
                        "result": [
                            {
                                "id": 1,
                                "orderId": 100,
                                "isBuyer": True,
                                "qty": "0.00038000",
                                "price": "73852.00",
                                "time": 1000,
                            },
                            {
                                "id": 2,
                                "orderId": 200,
                                "isBuyer": False,
                                "qty": "0.00019000",
                                "price": "64250.00",
                                "time": 2000,
                            },
                        ]
                    }
                return {"result": []}

            client._request_with_failover = fake_request
            reconciled = client.fetch_active_spot_holdings(("BTCUSDT", "SOLUSDT"))
            assert len(reconciled) == 1
            sym, rem_qty, rem_entry, has_partial = reconciled[0]
            assert sym == "BTCUSDT"
            assert rem_qty == pytest.approx(0.000190, abs=1e-8)
            assert rem_entry == pytest.approx(73_852.00, abs=1e-2)
            assert has_partial is True
        finally:
            ledger.close()


def test_qualified_altcoin_rotation_operates_when_btc_macro_is_bearish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify qualified altcoin rebound trades proceed during BEARISH BTC regime."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_bearish_rotation.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=51_600.0,
                    cop_per_usdt_rate=3_194.0,
                ),
                exchange=ExchangeConfig(symbol="XRPUSDT"),
                risk=replace(
                    RiskConfig.spot_no_loss_trailing_runner(btc_unlock_mode="NONE"),
                    enable_time_machine_gate=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(
                config=config.exchange,
                symbols=("BTCUSDT", "XRPUSDT", "LTCUSDT"),
            )
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)
            orchestrator.set_qualified_symbols({"XRPUSDT", "AVAXUSDT"})

            xrp_candles = [
                replace(c, symbol="XRPUSDT")
                for c in build_synthetic_candles(count=35, start_price=0.95, step=0.001)
            ]
            for c in xrp_candles[:-1]:
                orchestrator._upsert_candle(c)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: MarketSignal(
                    symbol="XRPUSDT",
                    evaluated_at=xrp_candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=xrp_candles[-1].close_price,
                    stop_loss_price=xrp_candles[-1].close_price * 0.995,
                    take_profit_price=xrp_candles[-1].close_price * 1.008,
                    atr_value=0.005,
                    rsi_value=49.5,
                    fast_ema=xrp_candles[-1].close_price * 0.999,
                    slow_ema=xrp_candles[-1].close_price,
                ),
            )
            res = orchestrator.process_candle_event(xrp_candles[-1])
            assert res.signal.signal_type == SignalType.BUY
            assert engine.has_position("XRPUSDT") is True
        finally:
            ledger.close()


def test_pre_cutoff_full_liquidation_sells_100pct_and_protects_new_buys() -> None:
    """Verify cutoff liquidation sells 100% of old coins and skips post-cutoff buys."""
    from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
    from crypto_paper_trader.notifications import format_full_reset_alert

    cutoff_ms = 1_795_113_720_000
    client = BinanceSpotAuthenticatedClient(
        config=ExchangeConfig(),
        api_key="k",
        api_secret="s",
    )
    client._step_size_cache["BTCUSDT"] = "0.00001000"
    client._step_size_cache["SUIUSDT"] = "0.10000000"
    client._step_size_cache["SOLUSDT"] = "0.00100000"
    client._step_size_cache["ETHUSDT"] = "0.00010000"

    price_map = {
        "BTCUSDT": 63_280.0,
        "SUIUSDT": 0.721,
        "SOLUSDT": 74.65,
        "ETHUSDT": 1_788.0,
    }
    client.fetch_latest_spot_price = lambda sym: price_map.get(sym.upper(), 0.0)

    sold_orders: list[tuple[str, float, str]] = []

    def fake_sell(
        symbol: str, quantity: float, client_order_id: str
    ) -> dict[str, object]:
        sold_orders.append((symbol, quantity, client_order_id))
        quote = round(quantity * price_map[symbol], 4)
        return {
            "executedQty": str(quantity),
            "cummulativeQuoteQty": str(quote),
        }

    client.place_market_sell_by_quantity = fake_sell

    def fake_request(
        method: str, path: str, params: dict[str, object]
    ) -> dict[str, object]:
        if path == "/api/v3/account":
            usdt_after = 27.95 if sold_orders else 0.90
            return {
                "balances": [
                    {"asset": "USDT", "free": str(usdt_after)},
                    {"asset": "BNB", "free": "0.00075000"},
                    {"asset": "BTC", "free": "0.00019961"},
                    {"asset": "SUI", "free": "10.50000000"},
                    {"asset": "SOL", "free": "0.10000000"},
                    {"asset": "ETH", "free": "0.00010000"},
                ]
            }
        if path == "/api/v3/myTrades":
            sym = str(params.get("symbol", ""))
            if sym == "BTCUSDT":
                return {
                    "result": [
                        {
                            "id": 1,
                            "orderId": 10,
                            "isBuyer": True,
                            "qty": "0.00038000",
                            "price": "73852.00",
                            "time": cutoff_ms - 500_000,
                        }
                    ]
                }
            if sym == "SUIUSDT":
                return {
                    "result": [
                        {
                            "id": 2,
                            "orderId": 11,
                            "isBuyer": True,
                            "qty": "10.50000000",
                            "price": "0.7258",
                            "time": cutoff_ms - 100_000,
                        }
                    ]
                }
            if sym == "SOLUSDT":
                return {
                    "result": [
                        {
                            "id": 3,
                            "orderId": 12,
                            "isBuyer": True,
                            "qty": "0.10000000",
                            "price": "74.65",
                            "time": cutoff_ms + 15_000,
                        }
                    ]
                }
        return {"result": []}

    client._request_with_failover = fake_request
    records = client.liquidate_pre_cutoff_spot_holdings(
        universe_symbols=("BTCUSDT", "SUIUSDT", "SOLUSDT", "ETHUSDT"),
        cutoff_epoch_ms=cutoff_ms,
    )
    sold_syms = [r["symbol"] for r in records]
    assert sold_syms == ["BTCUSDT", "SUIUSDT"]
    assert sold_orders[0][1] == pytest.approx(0.00019, abs=1e-8)
    assert sold_orders[1][1] == pytest.approx(10.5, abs=1e-8)
    assert client.get_known_usdt_balance() == pytest.approx(27.95, abs=1e-4)

    alert = format_full_reset_alert(
        liquidated_records=records,
        available_cash_usdt=27.95,
        cop_per_usdt_rate=3_194.0,
    )
    assert "LIQUIDACION 100% COMPLETADA" in alert
    assert "BTC" in alert and "SUI" in alert


def test_dynamic_4_slot_multicoin_fast_micro_scalp_and_60s_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 4 simultaneous slots split ~28 USDT and exit fast on micro rise."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_4slot_fast.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=89_432.0,
                    cop_per_usdt_rate=3_194.0,
                ),
                exchange=ExchangeConfig(symbol="SOLUSDT", taker_fee_rate=0.001),
                risk=replace(
                    RiskConfig.spot_fast_multicoin_micro_scalp(
                        min_target_net_profit_usdt=0.005,
                        max_concurrent_positions=4,
                    ),
                    enable_time_machine_gate=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(
                config=config.exchange,
                symbols=("SOLUSDT", "XRPUSDT", "SUIUSDT", "AVAXUSDT"),
            )
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )

            specs = [
                ("SOLUSDT", 74.0),
                ("XRPUSDT", 0.97),
                ("SUIUSDT", 0.72),
                ("AVAXUSDT", 7.10),
            ]
            last_candles: dict[str, Candle] = {}
            for sym, base_p in specs:
                sym_candles = [
                    replace(c, symbol=sym)
                    for c in build_synthetic_candles(
                        count=35, start_price=base_p, step=base_p * 0.001
                    )
                ]
                for c in sym_candles[:-1]:
                    orchestrator._upsert_candle(c)
                last_candles[sym] = sym_candles[-1]

            for sym, _ in specs:
                target_c = last_candles[sym]
                monkeypatch.setattr(
                    "crypto_paper_trader.orchestration.evaluate_market_signal",
                    lambda candles, symbol, config, tc=target_c: MarketSignal(
                        symbol=symbol,
                        evaluated_at=tc.close_time,
                        signal_type=SignalType.BUY,
                        reference_price=tc.close_price,
                        stop_loss_price=tc.close_price * 0.995,
                        take_profit_price=tc.close_price * 1.008,
                        atr_value=tc.close_price * 0.004,
                        rsi_value=45.0,
                        fast_ema=tc.close_price,
                        slow_ema=tc.close_price * 0.999,
                    ),
                )
                res = orchestrator.process_candle_event(target_c)
                assert res.signal.signal_type == SignalType.BUY

            assert len(engine.active_positions) == 4
            for pos in engine.active_positions:
                assert 5.5 <= pos.notional_usdt <= 8.0

            sol_pos = engine.get_open_position("SOLUSDT")
            assert sol_pos is not None
            sol_tp = sol_pos.take_profit_price
            exit_open = last_candles["SOLUSDT"].open_time + timedelta(minutes=2)
            exit_tick = Candle(
                symbol="SOLUSDT",
                open_time=exit_open,
                close_time=exit_open + timedelta(seconds=10),
                open_price=sol_pos.entry_price,
                high_price=sol_tp * 1.0005,
                low_price=sol_pos.entry_price,
                close_price=sol_tp * 1.0002,
                volume=500.0,
                is_closed=False,
            )
            orchestrator.process_candle_event(exit_tick)
            assert engine.has_position("SOLUSDT") is False
            assert len(engine.active_positions) == 3

            sell_execs = [
                e for e in ledger.fetch_all_executions() if e.side == OrderSide.SELL
            ]
            assert len(sell_execs) == 1
            assert sell_execs[0].reason == ExitReason.TAKE_PROFIT.value
            assert sell_execs[0].realized_pnl_usdt >= 0.005

            blocked_tick = Candle(
                symbol="SOLUSDT",
                open_time=exit_tick.open_time + timedelta(seconds=30),
                close_time=exit_tick.close_time + timedelta(seconds=30),
                open_price=sol_tp,
                high_price=sol_tp * 1.001,
                low_price=sol_tp * 0.999,
                close_price=sol_tp,
                volume=300.0,
                is_closed=False,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda candles, symbol, config: MarketSignal(
                    symbol=symbol,
                    evaluated_at=candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=candles[-1].close_price,
                    stop_loss_price=candles[-1].close_price * 0.995,
                    take_profit_price=candles[-1].close_price * 1.008,
                    atr_value=0.30,
                    rsi_value=45.0,
                    fast_ema=candles[-1].close_price,
                    slow_ema=candles[-1].close_price * 0.999,
                ),
            )
            res_blocked = orchestrator.process_candle_event(blocked_tick)
            assert res_blocked.signal.signal_type == SignalType.HOLD
            assert engine.has_position("SOLUSDT") is False

            reentry_tick = Candle(
                symbol="SOLUSDT",
                open_time=exit_tick.open_time + timedelta(seconds=65),
                close_time=exit_tick.close_time + timedelta(seconds=65),
                open_price=sol_tp,
                high_price=sol_tp * 1.001,
                low_price=sol_tp * 0.999,
                close_price=sol_tp,
                volume=300.0,
                is_closed=False,
            )
            res_reentry = orchestrator.process_candle_event(reentry_tick)
            assert res_reentry.signal.signal_type == SignalType.BUY
            assert engine.has_position("SOLUSDT") is True
            assert len(engine.active_positions) == 4
        finally:
            ledger.close()


def _build_daily_btc_series(closes: list[float], lows: list[float]) -> list[Candle]:
    """Build synthetic daily BTCUSDT candles for stabilization unit tests."""
    base_time = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for idx, (close_p, low_p) in enumerate(zip(closes, lows)):
        open_t = base_time + timedelta(days=idx)
        close_t = open_t + timedelta(hours=23, minutes=59)
        open_p = closes[idx - 1] if idx > 0 else close_p
        high_p = max(open_p, close_p) + 300.0
        low_val = min(low_p, open_p, close_p)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_t,
                close_time=close_t,
                open_price=open_p,
                high_price=high_p,
                low_price=low_val,
                close_price=close_p,
                volume=10_000.0,
                is_closed=True,
            )
        )
    return candles


def test_btc_stabilization_hysteresis_lock_unlock_and_relock() -> None:
    """Verify BTC lock stays LOCKED in downtrend, unlocks after 2 checks, relocks."""
    from crypto_paper_trader.market_intelligence import (
        BtcLockState,
        MarketIntelligenceService,
        evaluate_btc_stabilization,
    )

    down_closes = [75_000.0 - (idx * 650.0) for idx in range(25)]
    down_lows = [p - 400.0 for p in down_closes]
    down_candles = _build_daily_btc_series(down_closes, down_lows)

    state_down, reasons_down = evaluate_btc_stabilization(
        down_candles, previous_state=BtcLockState.LOCKED
    )
    assert state_down == BtcLockState.LOCKED
    assert len(reasons_down) >= 1

    rec_closes = [75_000.0 - (idx * 500.0) for idx in range(20)] + [
        65_200.0,
        66_000.0,
        67_500.0,
        68_800.0,
        69_500.0,
    ]
    rec_lows = [p - 300.0 for p in rec_closes]
    rec_lows[17] = 63_000.0
    rec_lows[-3:] = [65_500.0, 66_800.0, 67_900.0]
    rec_candles = _build_daily_btc_series(rec_closes, rec_lows)

    async def _run_service() -> None:
        async def fetch_rec() -> list[Candle]:
            return rec_candles

        async def fetch_fg() -> int:
            return 45

        async def fetch_mcap() -> float:
            return 1.2

        service = MarketIntelligenceService(
            exchange_config=ExchangeConfig(),
            mode="AUTO",
            daily_candles_fetcher=fetch_rec,
            fear_greed_fetcher=fetch_fg,
            global_mcap_fetcher=fetch_mcap,
        )
        first = await service.refresh(force=True)
        assert first.btc_lock_state == BtcLockState.LOCKED
        second = await service.refresh(force=True)
        assert second.btc_lock_state == BtcLockState.OPEN

    asyncio.run(_run_service())

    relock_closes = rec_closes + [61_000.0]
    relock_lows = rec_lows + [60_500.0]
    relock_candles = _build_daily_btc_series(relock_closes, relock_lows)
    state_relock, _ = evaluate_btc_stabilization(
        relock_candles, previous_state=BtcLockState.OPEN
    )
    assert state_relock == BtcLockState.LOCKED


def test_locked_btc_blocks_new_buys_and_dca_but_allows_altcoins_and_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify locked BTC blocks BTC BUY/DCA while altcoins and exits work normally."""
    from crypto_paper_trader.domain import Position
    from crypto_paper_trader.market_intelligence import (
        BtcLockState,
        MarketIntelligenceReport,
        MarketRiskLevel,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_btc_lock_exec.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
            btc_pos = Position(
                position_id="pos-btc-1",
                symbol="BTCUSDT",
                opened_at=t0 - timedelta(hours=12),
                entry_price=62_880.40,
                quantity=0.000100,
                notional_usdt=6.28804,
                entry_fee_usdt=0.0063,
                stop_loss_price=63_066.41,
                take_profit_price=63_082.18,
                peak_price=62_880.40,
            )
            ledger.save_open_position_for_symbol(btc_pos)

            config = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=90_000.0,
                    cop_per_usdt_rate=3_194.0,
                ),
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=replace(
                    RiskConfig.spot_fast_multicoin_micro_scalp(),
                    enable_time_machine_gate=False,
                    enable_dca_average_down=True,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(
                config=config.exchange,
                symbols=("BTCUSDT", "SOLUSDT"),
            )
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            intel_report = MarketIntelligenceReport(
                evaluated_at=t0,
                btc_lock_state=BtcLockState.LOCKED,
                risk_level=MarketRiskLevel.PANIC,
                btc_price=61_448.0,
                btc_daily_ema10=65_838.0,
                btc_daily_ema20=69_722.0,
                btc_change_24h_pct=-4.0,
                fear_greed_value=12,
                global_market_cap_change_24h_pct=-5.9,
                universe_breadth_pct=0.0,
                reasons=("BTC debajo de EMA10 diaria",),
            )
            orchestrator.set_market_intelligence(intel_report)

            assert engine.can_open_or_dca_symbol("BTCUSDT", 58_000.0) is False
            assert engine.can_open_position_for_symbol("SOLUSDT") is True

            btc_candles = [
                replace(c, symbol="BTCUSDT")
                for c in build_synthetic_candles(
                    count=35, start_price=60_000.0, step=10.0
                )
            ]
            for c in btc_candles[:-1]:
                orchestrator._upsert_candle(c)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: MarketSignal(
                    symbol="BTCUSDT",
                    evaluated_at=btc_candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=58_500.0,
                    stop_loss_price=58_000.0,
                    take_profit_price=59_000.0,
                    atr_value=300.0,
                    rsi_value=42.0,
                    fast_ema=58_500.0,
                    slow_ema=58_400.0,
                ),
            )
            btc_res = orchestrator.process_candle_event(btc_candles[-1])
            assert btc_res.signal.signal_type == SignalType.HOLD
            assert engine.get_open_position("BTCUSDT").quantity == pytest.approx(
                0.000100, abs=1e-8
            )

            sol_candles = [
                replace(c, symbol="SOLUSDT")
                for c in build_synthetic_candles(count=35, start_price=70.0, step=0.1)
            ]
            for c in sol_candles[:-1]:
                orchestrator._upsert_candle(c)
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: MarketSignal(
                    symbol="SOLUSDT",
                    evaluated_at=sol_candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=sol_candles[-1].close_price,
                    stop_loss_price=sol_candles[-1].close_price * 0.995,
                    take_profit_price=sol_candles[-1].close_price * 1.008,
                    atr_value=0.35,
                    rsi_value=45.0,
                    fast_ema=sol_candles[-1].close_price,
                    slow_ema=sol_candles[-1].close_price * 0.999,
                ),
            )
            sol_res = orchestrator.process_candle_event(sol_candles[-1])
            assert sol_res.signal.signal_type == SignalType.BUY
            assert engine.has_position("SOLUSDT") is True

            btc_tp_tick = Candle(
                symbol="BTCUSDT",
                open_time=btc_candles[-1].open_time + timedelta(minutes=15),
                close_time=btc_candles[-1].close_time + timedelta(minutes=15),
                open_price=63_000.0,
                high_price=63_120.0,
                low_price=62_990.0,
                close_price=63_100.0,
                volume=100.0,
                is_closed=False,
            )
            orchestrator.process_candle_event(btc_tp_tick)
            assert engine.has_position("BTCUSDT") is False
        finally:
            ledger.close()


def test_market_intelligence_failsafe_and_api_error_survival() -> None:
    """Verify service defaults to LOCKED without data and survives API errors."""
    from crypto_paper_trader.market_intelligence import (
        BtcLockState,
        MarketIntelligenceService,
    )

    async def _run() -> None:
        async def fail_daily() -> list[Candle]:
            raise RuntimeError("Network down")

        async def fail_fg() -> int | None:
            return None

        async def fail_mcap() -> float | None:
            return None

        service = MarketIntelligenceService(
            exchange_config=ExchangeConfig(),
            mode="AUTO",
            daily_candles_fetcher=fail_daily,
            fear_greed_fetcher=fail_fg,
            global_mcap_fetcher=fail_mcap,
        )
        report = await service.refresh(force=True)
        assert report.btc_lock_state == BtcLockState.LOCKED
        assert report.fear_greed_value is None
        assert report.global_market_cap_change_24h_pct is None

    asyncio.run(_run())


def test_force_modes_risk_classification_and_spanish_alerts() -> None:
    """Verify FORCE_LOCK, FORCE_OPEN, PANIC risk tier, and Spanish alerts."""
    from crypto_paper_trader.market_intelligence import (
        BtcLockState,
        MarketIntelligenceService,
        MarketRiskLevel,
        classify_market_risk,
    )
    from crypto_paper_trader.notifications import format_market_intelligence_alert

    assert classify_market_risk(12, 0.0) == MarketRiskLevel.PANIC
    assert classify_market_risk(30, 50.0) == MarketRiskLevel.CAUTION
    assert classify_market_risk(65, 70.0) == MarketRiskLevel.CONSTRUCTIVE

    down_closes = [75_000.0 - (idx * 650.0) for idx in range(25)]
    down_candles = _build_daily_btc_series(
        down_closes, [p - 400.0 for p in down_closes]
    )

    async def _run_overrides() -> None:
        async def fetch_down() -> list[Candle]:
            return down_candles

        async def fetch_fg() -> int:
            return 12

        async def fetch_mcap() -> float:
            return -5.9

        svc_open = MarketIntelligenceService(
            exchange_config=ExchangeConfig(),
            mode="FORCE_OPEN",
            daily_candles_fetcher=fetch_down,
            fear_greed_fetcher=fetch_fg,
            global_mcap_fetcher=fetch_mcap,
        )
        rep_open = await svc_open.refresh(force=True)
        assert rep_open.btc_lock_state == BtcLockState.OPEN
        assert rep_open.risk_level == MarketRiskLevel.PANIC

        svc_lock = MarketIntelligenceService(
            exchange_config=ExchangeConfig(),
            mode="FORCE_LOCK",
            daily_candles_fetcher=fetch_down,
            fear_greed_fetcher=fetch_fg,
            global_mcap_fetcher=fetch_mcap,
        )
        rep_lock = await svc_lock.refresh(force=True)
        assert rep_lock.btc_lock_state == BtcLockState.LOCKED

        alert_text = format_market_intelligence_alert(rep_lock)
        assert "INVESTIGACION DEL MERCADO CRIPTO" in alert_text
        assert "BLOQUEADO" in alert_text
        assert "PANICO EXTREMO" in alert_text
        assert "12/100" in alert_text

    asyncio.run(_run_overrides())

    multi_hb = format_heartbeat_alert(
        symbol="ETHUSDT",
        interval="15m",
        open_position_symbol="BTCUSDT + ETHUSDT",
        open_position_entry_price=62_880.40,
        available_cash_usdt=0.40,
        cop_per_usdt_rate=3_194.0,
        total_realized_pnl_usdt=0.0293,
        closed_trades_count=2,
        win_rate_pct=100.0,
        open_positions_detail=[("BTCUSDT", 62_880.40), ("ETHUSDT", 1_777.12)],
        market_intelligence_summary="Compras en BTC: BLOQUEADO hasta estabilizacion",
    )
    assert "BTC (62880.40 USDT), ETH (1777.12 USDT)" in multi_hb
    assert "BLOQUEADO" in multi_hb


def test_dual_target_asymmetric_slot_sizing_and_strict_zero_loss_floors() -> None:
    """Verify 1 SWING (~11.5 USDT, +1.40) + 2 SCALP (~7.3 USDT, +0.03) and no loss."""
    from crypto_paper_trader.domain import Position, SlotRole
    from crypto_paper_trader.execution import compute_no_loss_exit_levels

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_dual_target.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc)
            ada_sl, ada_tp = compute_no_loss_exit_levels(
                entry_price=0.1450,
                quantity=45.2,
                notional_usdt=6.5540,
                taker_fee_rate=0.001,
                min_target_net_profit_usdt=0.030,
            )
            ada_pos = Position(
                position_id="reconciled-adausdt",
                symbol="ADAUSDT",
                opened_at=t0 - timedelta(hours=6),
                entry_price=0.1450,
                quantity=45.2,
                notional_usdt=6.5540,
                entry_fee_usdt=0.0066,
                stop_loss_price=ada_sl,
                take_profit_price=ada_tp,
                peak_price=0.1450,
                slot_role=SlotRole.SCALP,
                target_net_profit_usdt=0.030,
                trough_price=0.1450,
            )
            ledger.save_open_position_for_symbol(ada_pos)

            risk_cfg = RiskConfig.spot_dual_scalp_and_swing(
                scalp_target_net_profit_usdt=0.030,
                swing_target_net_profit_usdt=1.40,
                max_concurrent_positions=3,
                swing_slot_count=1,
                swing_allocation_fraction=0.44,
            )
            capital_cfg = CapitalConfig(
                initial_capital_cop=92_056.0,
                cop_per_usdt_rate=3_485.0,
            )
            exchange_cfg = ExchangeConfig(symbol="SOLUSDT", taker_fee_rate=0.001)
            engine = PaperTradingEngine(
                capital_config=capital_cfg,
                exchange_config=exchange_cfg,
                risk_config=risk_cfg,
                ledger=ledger,
            )
            engine.set_swing_candidate_symbols({"LINKUSDT", "AVAXUSDT", "SUIUSDT"})

            assert engine.resolve_next_slot_role("LINKUSDT") == SlotRole.SWING
            assert engine.resolve_next_slot_role("SOLUSDT") == SlotRole.SCALP

            link_buy = MarketSignal(
                symbol="LINKUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=10.0,
                stop_loss_price=9.85,
                take_profit_price=10.15,
                atr_value=0.06,
                rsi_value=42.0,
                fast_ema=10.02,
                slow_ema=9.98,
            )
            link_exec = engine.process_signal(link_buy)
            assert link_exec is not None
            assert link_exec.client_order_id.startswith("swing-linkusdt-buy-")
            link_pos = engine.get_open_position("LINKUSDT")
            assert link_pos is not None
            assert link_pos.slot_role == SlotRole.SWING
            assert 11.0 <= link_pos.notional_usdt <= 12.2
            assert link_pos.stop_loss_price > link_pos.entry_price
            assert link_pos.take_profit_price > link_pos.entry_price * 1.11

            sol_buy = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0 + timedelta(minutes=1),
                signal_type=SignalType.BUY,
                reference_price=70.0,
                stop_loss_price=69.2,
                take_profit_price=70.8,
                atr_value=0.40,
                rsi_value=44.0,
                fast_ema=70.1,
                slow_ema=69.9,
            )
            sol_exec = engine.process_signal(sol_buy)
            assert sol_exec is not None
            assert sol_exec.client_order_id.startswith("scalp-solusdt-buy-")
            sol_pos = engine.get_open_position("SOLUSDT")
            assert sol_pos is not None
            assert sol_pos.slot_role == SlotRole.SCALP
            assert 6.5 <= sol_pos.notional_usdt <= 8.5
            assert sol_pos.take_profit_price < sol_pos.entry_price * 1.012

            assert engine.can_open_position_for_symbol("XRPUSDT") is False

            assert (
                engine.evaluate_protective_exits(
                    current_price=9.20,
                    timestamp=t0 + timedelta(minutes=10),
                    symbol="LINKUSDT",
                )
                is None
            )
            assert (
                engine.evaluate_protective_exits(
                    current_price=66.50,
                    timestamp=t0 + timedelta(minutes=10),
                    symbol="SOLUSDT",
                )
                is None
            )

            premature_swing_sell = MarketSignal(
                symbol="LINKUSDT",
                evaluated_at=t0 + timedelta(minutes=20),
                signal_type=SignalType.SELL,
                reference_price=10.18,
                stop_loss_price=9.90,
                take_profit_price=10.30,
                atr_value=0.06,
                rsi_value=66.0,
                fast_ema=10.12,
                slow_ema=10.05,
            )
            assert engine.process_signal(premature_swing_sell) is None
            assert engine.has_position("LINKUSDT") is True

            sol_tp_exec = engine.evaluate_protective_exits(
                current_price=sol_pos.take_profit_price + 0.01,
                timestamp=t0 + timedelta(minutes=25),
                symbol="SOLUSDT",
            )
            assert sol_tp_exec is not None
            assert sol_tp_exec.reason == ExitReason.TAKE_PROFIT.value
            assert sol_tp_exec.realized_pnl_usdt >= 0.030
            assert engine.has_position("SOLUSDT") is False

            link_tp_exec = engine.evaluate_protective_exits(
                current_price=link_pos.take_profit_price + 0.02,
                timestamp=t0 + timedelta(hours=18),
                symbol="LINKUSDT",
            )
            assert link_tp_exec is not None
            assert link_tp_exec.reason == ExitReason.TAKE_PROFIT.value
            assert link_tp_exec.realized_pnl_usdt >= 1.40
            assert engine.has_position("LINKUSDT") is False
            assert engine.has_position("ADAUSDT") is True
        finally:
            ledger.close()


def test_dual_target_swing_trailing_lock_and_optional_breakeven_escape() -> None:
    """Verify SWING progressive trailing stop locks gains and breakeven escape works."""
    from crypto_paper_trader.domain import SlotRole

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_swing_trail.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 22, 23, 30, tzinfo=timezone.utc)
            risk_cfg = RiskConfig.spot_dual_scalp_and_swing(
                scalp_target_net_profit_usdt=0.030,
                swing_target_net_profit_usdt=1.40,
                enable_swing_trailing_lock=True,
                swing_trailing_arm_ratio=0.065,
                swing_trailing_pullback_ratio=0.030,
                swing_trailing_min_lock_profit_usdt=0.40,
                enable_drawdown_breakeven_escape=True,
            )
            engine = PaperTradingEngine(
                capital_config=CapitalConfig(
                    initial_capital_cop=92_000.0,
                    cop_per_usdt_rate=3_485.0,
                ),
                exchange_config=ExchangeConfig(symbol="AVAXUSDT"),
                risk_config=risk_cfg,
                ledger=ledger,
            )
            engine.set_swing_candidate_symbols({"AVAXUSDT"})

            buy_sig = MarketSignal(
                symbol="AVAXUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=10.0,
                stop_loss_price=9.85,
                take_profit_price=10.20,
                atr_value=0.08,
                rsi_value=41.0,
                fast_ema=10.02,
                slow_ema=9.96,
            )
            buy_exec = engine.process_signal(buy_sig)
            assert buy_exec is not None
            pos = engine.get_open_position("AVAXUSDT")
            assert pos is not None
            assert pos.slot_role == SlotRole.SWING

            assert (
                engine.evaluate_protective_exits(
                    current_price=10.85,
                    timestamp=t0 + timedelta(hours=3),
                    symbol="AVAXUSDT",
                )
                is None
            )
            trail_exec = engine.evaluate_protective_exits(
                current_price=10.50,
                timestamp=t0 + timedelta(hours=4),
                symbol="AVAXUSDT",
            )
            assert trail_exec is not None
            assert trail_exec.reason == ExitReason.TRAILING_STOP.value
            assert trail_exec.realized_pnl_usdt >= 0.40

            buy_scalp = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=t0 + timedelta(hours=5),
                signal_type=SignalType.BUY,
                reference_price=0.1000,
                stop_loss_price=0.0985,
                take_profit_price=0.1015,
                atr_value=0.0008,
                rsi_value=43.0,
                fast_ema=0.1001,
                slow_ema=0.0998,
            )
            scalp_exec = engine.process_signal(buy_scalp)
            assert scalp_exec is not None
            doge_pos = engine.get_open_position("DOGEUSDT")
            assert doge_pos is not None
            assert doge_pos.slot_role == SlotRole.SCALP

            assert (
                engine.evaluate_protective_exits(
                    current_price=0.0955,
                    timestamp=t0 + timedelta(hours=6),
                    symbol="DOGEUSDT",
                )
                is None
            )
            below_be_sell = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=t0 + timedelta(hours=7),
                signal_type=SignalType.SELL,
                reference_price=0.0999,
                stop_loss_price=0.0980,
                take_profit_price=0.1020,
                atr_value=0.0008,
                rsi_value=63.0,
                fast_ema=0.0998,
                slow_ema=0.0996,
            )
            assert engine.process_signal(below_be_sell) is None

            be_sell = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=t0 + timedelta(hours=8),
                signal_type=SignalType.SELL,
                reference_price=0.10035,
                stop_loss_price=0.0980,
                take_profit_price=0.1020,
                atr_value=0.0008,
                rsi_value=64.0,
                fast_ema=0.1003,
                slow_ema=0.1001,
            )
            be_exec = engine.process_signal(be_sell)
            assert be_exec is not None
            assert be_exec.reason == ExitReason.BREAKEVEN_RECOVERY.value
            assert be_exec.realized_pnl_usdt >= 0.0
        finally:
            ledger.close()


def test_dual_target_persistence_and_render_restart_role_reconciliation() -> None:
    """Verify SQLite role persistence and deterministic Render restart inference."""
    from crypto_paper_trader.domain import Position, SlotRole
    from crypto_paper_trader.main import infer_reconciled_slot_roles

    risk_cfg = RiskConfig.spot_dual_scalp_and_swing(
        scalp_target_net_profit_usdt=0.030,
        swing_target_net_profit_usdt=1.40,
        max_concurrent_positions=3,
        swing_slot_count=1,
        swing_reconcile_min_notional_usdt=9.0,
    )

    ada_only_roles = infer_reconciled_slot_roles(
        holdings=[("ADAUSDT", 45.2, 0.1450, False)],
        free_usdt=19.8611,
        risk_config=risk_cfg,
    )
    assert ada_only_roles == {"ADAUSDT": SlotRole.SCALP}

    full_portfolio_roles = infer_reconciled_slot_roles(
        holdings=[
            ("ADAUSDT", 45.2, 0.1450, False),
            ("LINKUSDT", 1.30, 8.92, False),
            ("SOLUSDT", 0.103, 71.30, False),
        ],
        free_usdt=0.90,
        risk_config=risk_cfg,
    )
    assert full_portfolio_roles["LINKUSDT"] == SlotRole.SWING
    assert full_portfolio_roles["ADAUSDT"] == SlotRole.SCALP
    assert full_portfolio_roles["SOLUSDT"] == SlotRole.SCALP

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_role_persist.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            t0 = datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc)
            swing_pos = Position(
                position_id="pos-swing-1",
                symbol="LINKUSDT",
                opened_at=t0,
                entry_price=8.92,
                quantity=1.30,
                notional_usdt=11.596,
                entry_fee_usdt=0.0116,
                stop_loss_price=8.942,
                take_profit_price=10.02,
                peak_price=9.10,
                slot_role=SlotRole.SWING,
                target_net_profit_usdt=1.40,
                trough_price=8.85,
            )
            ledger.save_open_position_for_symbol(swing_pos)
            reloaded = ledger.fetch_all_open_positions()
            assert len(reloaded) == 1
            assert reloaded[0].slot_role == SlotRole.SWING
            assert reloaded[0].target_net_profit_usdt == pytest.approx(1.40)
            assert reloaded[0].trough_price == pytest.approx(8.85)
        finally:
            ledger.close()


def test_dual_target_scanner_candidates_time_machine_and_alerts() -> None:
    """Verify Swing candidate selection, Time Machine dual role, and ntfy alerts."""
    from crypto_paper_trader.domain import SlotRole
    from crypto_paper_trader.scanner import (
        AssetProjectionScore,
        select_swing_candidate_symbols,
    )
    from crypto_paper_trader.simulation import run_time_machine_simulation

    scores = [
        AssetProjectionScore(
            "BTCUSDT", "STABLE_CORE", 61500.0, 0.46, 0.1, 80.0, 2.0, 4, 65.0
        ),
        AssetProjectionScore(
            "LINKUSDT", "HIGH_VOLATILITY", 8.88, 0.68, 0.1, 90.0, 3.0, 4, 59.0
        ),
        AssetProjectionScore(
            "AVAXUSDT", "HIGH_VOLATILITY", 6.68, 0.65, 0.1, 85.0, 2.5, 4, 58.5
        ),
        AssetProjectionScore(
            "LTCUSDT", "STABLE_CORE", 40.3, 0.40, 0.05, 70.0, 1.5, 3, 52.0
        ),
    ]
    swing_picks = select_swing_candidate_symbols(
        rankings=scores,
        blocked_symbols={"BTCUSDT"},
        max_symbols=2,
    )
    assert "BTCUSDT" not in swing_picks
    assert swing_picks == {"LINKUSDT", "AVAXUSDT"}

    candles = build_synthetic_candles(count=60, start_price=10.0, step=0.05)
    dual_risk = RiskConfig.spot_dual_scalp_and_swing()
    now = datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc)
    good_sig = MarketSignal(
        symbol="LINKUSDT",
        evaluated_at=now,
        signal_type=SignalType.BUY,
        reference_price=candles[-1].close_price,
        stop_loss_price=candles[-1].close_price * 0.99,
        take_profit_price=candles[-1].close_price * 1.02,
        atr_value=0.10,
        rsi_value=45.0,
        fast_ema=candles[-1].close_price,
        slow_ema=candles[-1].close_price * 0.997,
    )
    swing_tm = run_time_machine_simulation(
        candles=candles,
        signal=good_sig,
        risk_config=dual_risk,
        exchange_config=ExchangeConfig(symbol="LINKUSDT"),
        replays_count=4_000,
        min_win_probability_pct=52.0,
        target_role=SlotRole.SWING,
    )
    assert swing_tm.approved is True

    sell_exec = TradeExecution(
        client_order_id="scalp-solusdt-sell-abc",
        position_id="pos-sol-1",
        symbol="SOLUSDT",
        side=OrderSide.SELL,
        executed_at=now,
        execution_price=71.80,
        quantity=0.103,
        gross_notional_usdt=7.3954,
        fee_usdt=0.0074,
        realized_pnl_usdt=0.0320,
        reason=ExitReason.TAKE_PROFIT.value,
    )
    sell_alert = format_execution_alert(
        execution=sell_exec,
        available_cash_usdt=8.25,
        cop_per_usdt_rate=3_485.0,
        is_live_order=True,
        slot_role=SlotRole.SCALP,
        target_net_profit_usdt=0.030,
        open_positions_summary=[
            ("ADAUSDT", 6.55, "SCALP"),
            ("LINKUSDT", 11.60, "SWING"),
        ],
    )
    assert "[SCALP ~105 COP]" in sell_alert
    assert "ADA [SCALP], LINK [SWING]" in sell_alert
    assert "Monedas abiertas: Ninguna" not in sell_alert
    assert "BALANCE TOTAL: 26.40 USDT" in sell_alert
    assert translate_exit_reason(ExitReason.BREAKEVEN_RECOVERY.value) != ""


def test_universal_3d_stabilization_lock_across_all_symbols_and_orchestrator() -> None:
    """Verify strict 3-day no-drop + EMA10 lock applies independently to all coins."""
    from crypto_paper_trader.market_intelligence import (
        BtcLockState,
        MarketIntelligenceService,
        evaluate_btc_stabilization,
        evaluate_symbol_stabilization,
    )
    from crypto_paper_trader.notifications import format_market_intelligence_alert

    base_closes = [72_000.0 - (idx * 550.0) for idx in range(18)]
    base_lows = [p - 400.0 for p in base_closes]
    bounce_below_ema10_closes = base_closes + [
        61_986.7,
        58_951.5,
        60_215.8,
        61_271.1,
        61_480.0,
    ]
    bounce_below_ema10_lows = base_lows + [
        61_428.3,
        57_778.3,
        55_130.0,
        58_768.5,
        60_386.0,
    ]
    btc_below_ema10 = _build_daily_btc_series(
        bounce_below_ema10_closes, bounce_below_ema10_lows
    )
    state_btc, _ = evaluate_btc_stabilization(
        btc_below_ema10, previous_state=BtcLockState.LOCKED, repaired_streak=2
    )
    assert state_btc == BtcLockState.LOCKED

    healthy_closes = [100.0 - (idx * 0.8) for idx in range(18)] + [
        88.0,
        90.0,
        93.0,
        96.0,
        98.5,
    ]
    healthy_lows = [p - 0.5 for p in healthy_closes]
    healthy_lows[15] = 82.0
    healthy_lows[-3:] = [91.0, 94.0, 96.5]
    sol_healthy = [
        replace(c, symbol="SOLUSDT")
        for c in _build_daily_btc_series(healthy_closes, healthy_lows)
    ]
    wif_healthy = [
        replace(c, symbol="WIFUSDT")
        for c in _build_daily_btc_series(healthy_closes, healthy_lows)
    ]
    eth_weak = [replace(c, symbol="ETHUSDT") for c in btc_below_ema10]

    state_sol_open, _ = evaluate_symbol_stabilization(
        "SOLUSDT",
        sol_healthy,
        previous_state=BtcLockState.LOCKED,
        repaired_streak=2,
    )
    assert state_sol_open == BtcLockState.OPEN

    series_map = {
        "BTCUSDT": btc_below_ema10,
        "ETHUSDT": eth_weak,
        "SOLUSDT": sol_healthy,
        "WIFUSDT": wif_healthy,
    }

    async def _run_multi() -> None:
        async def fetch_by_sym(sym: str) -> list[Candle]:
            return series_map[sym]

        async def fetch_fg() -> int:
            return 25

        async def fetch_mcap() -> float:
            return 0.5

        svc = MarketIntelligenceService(
            exchange_config=ExchangeConfig(),
            mode="AUTO",
            tracked_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "WIFUSDT"),
            daily_candles_by_symbol_fetcher=fetch_by_sym,
            fear_greed_fetcher=fetch_fg,
            global_mcap_fetcher=fetch_mcap,
        )
        rep1 = await svc.refresh(force=True)
        assert rep1.unlocked_symbols == ()
        assert set(rep1.locked_symbols) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "WIFUSDT"}

        rep2 = await svc.refresh(force=False)
        assert rep2.btc_lock_state == BtcLockState.LOCKED
        assert rep2.unlocked_symbols == ("SOLUSDT", "WIFUSDT")
        assert rep2.locked_symbols == ("BTCUSDT", "ETHUSDT")

        alert_txt = format_market_intelligence_alert(rep2)
        assert "Candado 3 dias sin caidas (todas las monedas): 2/4" in alert_txt
        assert "SOL, WIF" in alert_txt

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "universal_lock.db"
            ledger = LedgerRepository(db_path=db_path)
            try:
                cfg = TradingSystemConfig(
                    risk=RiskConfig.spot_dual_scalp_and_swing(
                        enable_universal_3d_stabilization_lock=True
                    ),
                    ledger_db_path=db_path,
                )
                engine = PaperTradingEngine(
                    capital_config=cfg.capital,
                    exchange_config=cfg.exchange,
                    risk_config=cfg.risk,
                    ledger=ledger,
                )
                provider = BinanceMarketDataClient(
                    config=cfg.exchange,
                    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "WIFUSDT"),
                )
                orch = RealTimeTradingOrchestrator(
                    config=cfg,
                    market_data_provider=provider,
                    trading_engine=engine,
                    multi_symbol_mode=True,
                )
                orch.set_market_intelligence(rep2)
                assert engine.can_open_position_for_symbol("BTCUSDT") is False
                assert engine.can_open_position_for_symbol("ETHUSDT") is False
                assert engine.can_open_position_for_symbol("SOLUSDT") is True
                assert engine.can_open_position_for_symbol("WIFUSDT") is True
            finally:
                ledger.close()

    asyncio.run(_run_multi())


def test_scalp_trailing_runner_swing_exclusion_and_btc_reclassification() -> None:
    """Verify slow-coin Swing exclusion, BTC SCALP reclassification, and runner."""
    from crypto_paper_trader.domain import SlotRole
    from crypto_paper_trader.main import infer_reconciled_slot_roles
    from crypto_paper_trader.scanner import (
        AssetProjectionScore,
        select_swing_candidate_symbols,
    )

    risk_cfg = RiskConfig.spot_dual_scalp_and_swing(
        scalp_target_net_profit_usdt=0.030,
        swing_target_net_profit_usdt=0.55,
        enable_scalp_trailing_runner=True,
        scalp_trailing_arm_ratio=0.0048,
        scalp_trailing_pullback_ratio=0.0035,
        swing_excluded_symbols=("BTCUSDT", "ETHUSDT", "LTCUSDT"),
    )

    reconciled_nov26 = infer_reconciled_slot_roles(
        holdings=[
            ("BTCUSDT", 0.000190, 61_204.01, False),
            ("ETHUSDT", 0.004300, 1_744.30, False),
            ("AVAXUSDT", 1.0700, 6.8140, False),
        ],
        free_usdt=0.42,
        risk_config=risk_cfg,
    )
    assert reconciled_nov26["BTCUSDT"] == SlotRole.SCALP
    assert reconciled_nov26["ETHUSDT"] == SlotRole.SCALP
    assert reconciled_nov26["AVAXUSDT"] == SlotRole.SCALP

    reconciled_alt_swing = infer_reconciled_slot_roles(
        holdings=[
            ("SOLUSDT", 0.165, 71.0, False),
            ("ADAUSDT", 45.2, 0.145, False),
        ],
        free_usdt=8.0,
        risk_config=risk_cfg,
    )
    assert reconciled_alt_swing["SOLUSDT"] == SlotRole.SWING
    assert reconciled_alt_swing["ADAUSDT"] == SlotRole.SCALP

    scores = [
        AssetProjectionScore(
            "BTCUSDT", "STABLE_CORE", 60500.0, 0.35, 0.2, 100.0, 99.9, 5, 72.0
        ),
        AssetProjectionScore(
            "ETHUSDT", "STABLE_CORE", 1690.0, 0.41, 0.2, 100.0, 99.9, 5, 65.0
        ),
        AssetProjectionScore(
            "WIFUSDT", "HIGH_VOLATILITY", 0.14, 0.72, 0.5, 100.0, 99.9, 5, 62.0
        ),
        AssetProjectionScore(
            "NEARUSDT", "HIGH_VOLATILITY", 3.08, 0.85, 0.4, 100.0, 99.9, 4, 60.0
        ),
    ]
    picks = select_swing_candidate_symbols(
        rankings=scores,
        blocked_symbols=set(),
        max_symbols=3,
        excluded_symbols=("BTCUSDT", "ETHUSDT", "LTCUSDT"),
    )
    assert picks == {"WIFUSDT", "NEARUSDT"}

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "runner_test.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            engine = PaperTradingEngine(
                capital_config=CapitalConfig(
                    initial_capital_cop=92_600.0, cop_per_usdt_rate=3_485.0
                ),
                exchange_config=ExchangeConfig(symbol="SOLUSDT", taker_fee_rate=0.001),
                risk_config=risk_cfg,
                ledger=ledger,
            )
            assert engine.resolve_next_slot_role("BTCUSDT") == SlotRole.SCALP
            assert engine.resolve_next_slot_role("WIFUSDT") == SlotRole.SWING

            t0 = datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc)
            btc_buy = MarketSignal(
                symbol="BTCUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=60_000.0,
                stop_loss_price=59_600.0,
                take_profit_price=60_500.0,
                atr_value=200.0,
                rsi_value=44.0,
                fast_ema=60_050.0,
                slow_ema=59_950.0,
            )
            buy_exec = engine.process_signal(btc_buy)
            assert buy_exec is not None
            pos = engine.get_open_position("BTCUSDT")
            assert pos is not None
            assert pos.slot_role == SlotRole.SCALP

            assert (
                engine.evaluate_protective_exits(
                    current_price=61_320.0,
                    timestamp=t0 + timedelta(minutes=15),
                    symbol="BTCUSDT",
                )
                is None
            )
            early_rev = MarketSignal(
                symbol="BTCUSDT",
                evaluated_at=t0 + timedelta(minutes=16),
                signal_type=SignalType.SELL,
                reference_price=61_250.0,
                stop_loss_price=60_000.0,
                take_profit_price=62_000.0,
                atr_value=200.0,
                rsi_value=66.0,
                fast_ema=61_100.0,
                slow_ema=60_800.0,
            )
            assert engine.process_signal(early_rev) is None

            trail_exec = engine.evaluate_protective_exits(
                current_price=61_080.0,
                timestamp=t0 + timedelta(minutes=20),
                symbol="BTCUSDT",
            )
            assert trail_exec is not None
            assert trail_exec.reason == ExitReason.TRAILING_STOP.value
            assert trail_exec.realized_pnl_usdt > 0.10
        finally:
            ledger.close()


def test_anti_trap_24h_filter_stagger_and_held_buy_silence(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify 24h/1h anti-trap filter, portfolio stagger, and quiet held BUY ticks."""
    import logging

    overextended = build_synthetic_candles(count=96, start_price=100.0, step=0.25)
    passed_top, reason_top = RealTimeTradingOrchestrator._evaluate_anti_trap_24h_filter(
        overextended, max_range_position_ratio=0.72
    )
    assert passed_top is False
    assert "24h high" in reason_top

    base_t = datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)
    falling_knife: list[Candle] = []
    for idx in range(96):
        price = 100.0 + (idx * 0.2 if idx < 50 else (10.0 - (idx - 50) * 0.14))
        falling_knife.append(
            Candle(
                symbol="SOLUSDT",
                open_time=base_t + timedelta(minutes=15 * idx),
                close_time=base_t + timedelta(minutes=15 * idx + 14),
                open_price=price - 0.1,
                high_price=price + 0.3,
                low_price=price - 0.3,
                close_price=price,
                volume=100.0,
                is_closed=True,
            )
        )
    passed_knife, reason_knife = (
        RealTimeTradingOrchestrator._evaluate_anti_trap_24h_filter(
            falling_knife, max_range_position_ratio=0.72
        )
    )
    assert passed_knife is False
    assert "EMA21" in reason_knife

    healthy_pullback: list[Candle] = []
    for idx in range(96):
        price = 100.0 + (idx * 0.10 if idx < 80 else (8.0 - (idx - 80) * 0.18))
        healthy_pullback.append(
            Candle(
                symbol="SOLUSDT",
                open_time=base_t + timedelta(minutes=15 * idx),
                close_time=base_t + timedelta(minutes=15 * idx + 14),
                open_price=price - 0.1,
                high_price=price + 0.3,
                low_price=price - 0.3,
                close_price=price,
                volume=100.0,
                is_closed=True,
            )
        )
    passed_pb, _ = RealTimeTradingOrchestrator._evaluate_anti_trap_24h_filter(
        healthy_pullback, max_range_position_ratio=0.72
    )
    assert passed_pb is True

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "stagger_test.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            cfg = TradingSystemConfig(
                capital=CapitalConfig(
                    initial_capital_cop=92_600.0, cop_per_usdt_rate=3_485.0
                ),
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=replace(
                    RiskConfig.spot_dual_scalp_and_swing(
                        portfolio_entry_stagger_seconds=2700.0,
                        enable_anti_trap_24h_filter=False,
                    ),
                    enable_time_machine_gate=False,
                ),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=cfg.capital,
                exchange_config=cfg.exchange,
                risk_config=cfg.risk,
                ledger=ledger,
            )
            provider = BinanceMarketDataClient(
                config=cfg.exchange, symbols=("SOLUSDT", "XRPUSDT")
            )
            orch = RealTimeTradingOrchestrator(
                config=cfg,
                market_data_provider=provider,
                trading_engine=engine,
                multi_symbol_mode=True,
            )
            sol_cs = [
                replace(c, symbol="SOLUSDT")
                for c in build_synthetic_candles(count=35, start_price=70.0, step=0.1)
            ]
            xrp_cs = [
                replace(c, symbol="XRPUSDT")
                for c in build_synthetic_candles(count=35, start_price=0.95, step=0.002)
            ]
            for c in sol_cs[:-1]:
                orch._upsert_candle(c)
            for c in xrp_cs[:-1]:
                orch._upsert_candle(c)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda candles, symbol, config: MarketSignal(
                    symbol=symbol,
                    evaluated_at=candles[-1].close_time,
                    signal_type=SignalType.BUY,
                    reference_price=candles[-1].close_price,
                    stop_loss_price=candles[-1].close_price * 0.992,
                    take_profit_price=candles[-1].close_price * 1.01,
                    atr_value=candles[-1].close_price * 0.005,
                    rsi_value=45.0,
                    fast_ema=candles[-1].close_price,
                    slow_ema=candles[-1].close_price * 0.999,
                ),
            )
            orch.process_candle_event(sol_cs[-1])
            assert engine.has_position("SOLUSDT") is True

            soon_xrp = replace(
                xrp_cs[-1],
                open_time=sol_cs[-1].open_time + timedelta(minutes=5),
                close_time=sol_cs[-1].close_time + timedelta(minutes=5),
            )
            orch.process_candle_event(soon_xrp)
            assert engine.has_position("XRPUSDT") is False

            with caplog.at_level(logging.INFO):
                caplog.clear()
                held_sol_tick = replace(
                    sol_cs[-1],
                    open_time=sol_cs[-1].open_time + timedelta(minutes=10),
                    close_time=sol_cs[-1].close_time + timedelta(minutes=10),
                )
                orch.process_candle_event(held_sol_tick)
                assert "Suppressed BUY" not in caplog.text

            later_xrp = replace(
                xrp_cs[-1],
                open_time=sol_cs[-1].open_time + timedelta(minutes=50),
                close_time=sol_cs[-1].close_time + timedelta(minutes=50),
            )
            orch.process_candle_event(later_xrp)
            assert engine.has_position("XRPUSDT") is True
        finally:
            ledger.close()
