from __future__ import annotations

import tempfile
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
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.live_exchange import quantize_step_size
from crypto_paper_trader.notifications import format_execution_alert
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
