from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crypto_paper_trader.config import (
    ExchangeConfig,
    RiskConfig,
    TradingSystemConfig,
)
from crypto_paper_trader.domain import (
    Candle,
    ExitReason,
    MacroMarketRegime,
    MarketSignal,
    SignalType,
)
from crypto_paper_trader.execution import LedgerRepository, PaperTradingEngine
from crypto_paper_trader.orchestration import RealTimeTradingOrchestrator


def build_synthetic_candle(
    symbol: str,
    open_time: datetime,
    duration_seconds: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float = 100.0,
) -> Candle:
    """Construct an immutable Candle instance for testing.

    Args:
        symbol: Trading pair identifier.
        open_time: UTC timestamp for candle open.
        duration_seconds: Duration of the candle in seconds.
        open_price: Opening price.
        high_price: High price.
        low_price: Low price.
        close_price: Close price.
        volume: Volume traded.

    Returns:
        A populated Candle instance.
    """
    close_time = open_time + timedelta(seconds=duration_seconds - 1)
    return Candle(
        symbol=symbol,
        open_time=open_time,
        close_time=close_time,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        is_closed=True,
    )


def bootstrap_orchestrator(
    orchestrator: RealTimeTradingOrchestrator,
    start_time: datetime,
    base_price: float = 100.0,
    count: int = 35,
) -> None:
    """Pre-populate orchestrator candle buffer with historical candles.

    Args:
        orchestrator: RealTimeTradingOrchestrator instance to prime.
        start_time: Starting UTC timestamp.
        base_price: Baseline price for synthetic candles.
        count: Number of historical candles to insert.

    Returns:
        None.
    """
    for idx in range(count):
        candle_open = start_time + timedelta(minutes=idx)
        candle = build_synthetic_candle(
            symbol=orchestrator._config.exchange.symbol,
            open_time=candle_open,
            duration_seconds=60,
            open_price=round(base_price, 6),
            high_price=round(base_price * 1.001, 6),
            low_price=round(base_price * 0.999, 6),
            close_price=round(base_price, 6),
            volume=500.0,
        )
        orchestrator.process_candle_event(candle)


def test_cooldown_strictly_blocks_reentries_after_stop_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a 900s cooldown strictly suppresses BUY signals after STOP_LOSS.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_sl_cooldown.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="DOGEUSDT"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            t0 = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=0.1000, count=35)

            open_candle_time = t0 + timedelta(minutes=35)
            buy_signal_template = MarketSignal(
                symbol="DOGEUSDT",
                evaluated_at=open_candle_time,
                signal_type=SignalType.BUY,
                reference_price=0.1000,
                stop_loss_price=0.0995,
                take_profit_price=0.1020,
                atr_value=0.0005,
                rsi_value=48.0,
                fast_ema=0.1001,
                slow_ema=0.0999,
            )

            def mock_evaluator(candles, **kwargs):
                return replace(buy_signal_template, evaluated_at=candles[-1].close_time)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_evaluator,
            )
            entry_candle = build_synthetic_candle(
                symbol="DOGEUSDT",
                open_time=open_candle_time,
                duration_seconds=60,
                open_price=0.1000,
                high_price=0.1001,
                low_price=0.0999,
                close_price=0.1000,
            )
            orchestrator.process_candle_event(entry_candle)
            assert engine.active_position is not None

            sl_candle_time = t0 + timedelta(minutes=37)
            sl_candle = build_synthetic_candle(
                symbol="DOGEUSDT",
                open_time=sl_candle_time,
                duration_seconds=60,
                open_price=0.0998,
                high_price=0.0998,
                low_price=0.0993,
                close_price=0.0994,
            )
            orchestrator.process_candle_event(sl_candle)
            assert engine.active_position is None
            expected_sl_cooldown = sl_candle.close_time.timestamp() + 900.0
            assert orchestrator._cooldown_until_ts == expected_sl_cooldown

            noise_offsets_seconds = [1, 15, 60, 180, 300, 500, 899]
            for offset in noise_offsets_seconds:
                tick_open = sl_candle.close_time + timedelta(seconds=offset - 1)
                noise_candle = build_synthetic_candle(
                    symbol="DOGEUSDT",
                    open_time=tick_open,
                    duration_seconds=1,
                    open_price=0.1000,
                    high_price=0.1005,
                    low_price=0.0999,
                    close_price=0.1002,
                )
                tick_result = orchestrator.process_candle_event(noise_candle)
                assert (
                    tick_result.signal.signal_type == SignalType.HOLD
                ), f"Failed at offset {offset}s within 900s cooldown"
                assert engine.active_position is None

            unblock_open = sl_candle.close_time + timedelta(seconds=901)
            unblock_candle = build_synthetic_candle(
                symbol="DOGEUSDT",
                open_time=unblock_open,
                duration_seconds=60,
                open_price=0.1000,
                high_price=0.1002,
                low_price=0.0998,
                close_price=0.1000,
            )
            unblock_result = orchestrator.process_candle_event(unblock_candle)
            assert unblock_result.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None
        finally:
            ledger.close()


def test_cooldown_strictly_blocks_reentries_after_trailing_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a 300s cooldown strictly suppresses BUY after TRAILING_STOP.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_ts_cooldown.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="XRPUSDT"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            t0 = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=1.3000, count=35)

            open_time = t0 + timedelta(minutes=35)
            buy_signal_template = MarketSignal(
                symbol="XRPUSDT",
                evaluated_at=open_time,
                signal_type=SignalType.BUY,
                reference_price=1.3000,
                stop_loss_price=1.2940,
                take_profit_price=1.3200,
                atr_value=0.0050,
                rsi_value=50.0,
                fast_ema=1.3010,
                slow_ema=1.2990,
            )

            def mock_evaluator(candles, **kwargs):
                return replace(buy_signal_template, evaluated_at=candles[-1].close_time)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_evaluator,
            )
            entry_candle = build_synthetic_candle(
                symbol="XRPUSDT",
                open_time=open_time,
                duration_seconds=60,
                open_price=1.3000,
                high_price=1.3010,
                low_price=1.2990,
                close_price=1.3000,
            )
            orchestrator.process_candle_event(entry_candle)
            assert engine.active_position is not None

            pump_time = t0 + timedelta(minutes=36)
            pump_candle = build_synthetic_candle(
                symbol="XRPUSDT",
                open_time=pump_time,
                duration_seconds=60,
                open_price=1.3000,
                high_price=1.3050,
                low_price=1.3000,
                close_price=1.3045,
            )
            orchestrator.process_candle_event(pump_candle)
            active_pos = engine.active_position
            assert active_pos is not None
            assert active_pos.stop_loss_price >= 1.3000

            dump_time = t0 + timedelta(minutes=37)
            dump_candle = build_synthetic_candle(
                symbol="XRPUSDT",
                open_time=dump_time,
                duration_seconds=60,
                open_price=1.3030,
                high_price=1.3030,
                low_price=1.3005,
                close_price=1.3010,
            )
            orchestrator.process_candle_event(dump_candle)
            assert engine.active_position is None
            expected_ts_cooldown = dump_candle.close_time.timestamp() + 300.0
            assert orchestrator._cooldown_until_ts == expected_ts_cooldown

            noise_offsets = [1, 10, 60, 150, 299]
            for offset in noise_offsets:
                tick_time = dump_candle.close_time + timedelta(seconds=offset - 1)
                noise_candle = build_synthetic_candle(
                    symbol="XRPUSDT",
                    open_time=tick_time,
                    duration_seconds=1,
                    open_price=1.3020,
                    high_price=1.3030,
                    low_price=1.3010,
                    close_price=1.3025,
                )
                res = orchestrator.process_candle_event(noise_candle)
                assert (
                    res.signal.signal_type == SignalType.HOLD
                ), f"Failed at offset {offset}s within 300s trailing stop cooldown"
                assert engine.active_position is None

            unblock_time = dump_candle.close_time + timedelta(seconds=301)
            unblock_candle = build_synthetic_candle(
                symbol="XRPUSDT",
                open_time=unblock_time,
                duration_seconds=60,
                open_price=1.3000,
                high_price=1.3010,
                low_price=1.2990,
                close_price=1.3000,
            )
            res_unblock = orchestrator.process_candle_event(unblock_candle)
            assert res_unblock.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None
        finally:
            ledger.close()


def test_cooldown_strictly_blocks_reentries_after_take_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a 300s cooldown strictly suppresses BUY after TAKE_PROFIT.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tp_cooldown.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                risk=RiskConfig(enable_partial_take_profit=False),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            t0 = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=100.0, count=35)

            open_time = t0 + timedelta(minutes=35)
            buy_signal_template = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=open_time,
                signal_type=SignalType.BUY,
                reference_price=100.0,
                stop_loss_price=99.5,
                take_profit_price=101.0,
                atr_value=0.5,
                rsi_value=50.0,
                fast_ema=100.2,
                slow_ema=99.8,
            )

            def mock_evaluator(candles, **kwargs):
                return replace(buy_signal_template, evaluated_at=candles[-1].close_time)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_evaluator,
            )
            entry_candle = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=open_time,
                duration_seconds=60,
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.0,
            )
            orchestrator.process_candle_event(entry_candle)
            assert engine.active_position is not None

            tp_price = engine.active_position.take_profit_price
            tp_time = t0 + timedelta(minutes=38)
            tp_candle = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=tp_time,
                duration_seconds=60,
                open_price=tp_price - 0.1,
                high_price=tp_price + 0.5,
                low_price=tp_price - 0.2,
                close_price=tp_price + 0.1,
            )
            orchestrator.process_candle_event(tp_candle)
            assert engine.active_position is None
            expected_tp_cooldown = tp_candle.close_time.timestamp() + 300.0
            assert orchestrator._cooldown_until_ts == expected_tp_cooldown

            for offset in [5, 50, 150, 299]:
                noise_time = tp_candle.close_time + timedelta(seconds=offset - 1)
                noise_candle = build_synthetic_candle(
                    symbol="SOLUSDT",
                    open_time=noise_time,
                    duration_seconds=1,
                    open_price=100.0,
                    high_price=100.5,
                    low_price=99.9,
                    close_price=100.1,
                )
                res = orchestrator.process_candle_event(noise_candle)
                assert res.signal.signal_type == SignalType.HOLD
                assert engine.active_position is None

            unblock_time = tp_candle.close_time + timedelta(seconds=301)
            unblock_candle = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=unblock_time,
                duration_seconds=60,
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.0,
            )
            res_unblock = orchestrator.process_candle_event(unblock_candle)
            assert res_unblock.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None
        finally:
            ledger.close()


def test_cooldown_strictly_blocks_reentries_after_signal_reversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a 300s cooldown strictly suppresses BUY after SIGNAL_REVERSAL.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_rev_cooldown.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="AVAXUSDT"),
                risk=RiskConfig(enable_partial_take_profit=False),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            t0 = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=10.0, count=35)

            open_time = t0 + timedelta(minutes=35)
            buy_signal_template = MarketSignal(
                symbol="AVAXUSDT",
                evaluated_at=open_time,
                signal_type=SignalType.BUY,
                reference_price=10.00,
                stop_loss_price=9.95,
                take_profit_price=10.40,
                atr_value=0.05,
                rsi_value=49.0,
                fast_ema=10.02,
                slow_ema=9.98,
            )

            def mock_buy_evaluator(candles, **kwargs):
                return replace(buy_signal_template, evaluated_at=candles[-1].close_time)

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_buy_evaluator,
            )
            entry_candle = build_synthetic_candle(
                symbol="AVAXUSDT",
                open_time=open_time,
                duration_seconds=60,
                open_price=10.00,
                high_price=10.02,
                low_price=9.98,
                close_price=10.00,
            )
            orchestrator.process_candle_event(entry_candle)
            assert engine.active_position is not None

            rev_time = t0 + timedelta(minutes=40)
            rev_signal = MarketSignal(
                symbol="AVAXUSDT",
                evaluated_at=rev_time,
                signal_type=SignalType.SELL,
                reference_price=10.035,
                stop_loss_price=9.95,
                take_profit_price=10.40,
                atr_value=0.05,
                rsi_value=59.5,
                fast_ema=10.03,
                slow_ema=9.99,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: rev_signal,
            )
            rev_candle = build_synthetic_candle(
                symbol="AVAXUSDT",
                open_time=rev_time,
                duration_seconds=60,
                open_price=10.03,
                high_price=10.04,
                low_price=10.02,
                close_price=10.035,
            )
            rev_result = orchestrator.process_candle_event(rev_candle)
            assert rev_result.signal.signal_type == SignalType.SELL
            assert engine.active_position is None
            expected_rev_cooldown = rev_candle.close_time.timestamp() + 300.0
            assert orchestrator._cooldown_until_ts == expected_rev_cooldown

            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                mock_buy_evaluator,
            )
            for offset in [2, 30, 100, 200, 299]:
                noise_time = rev_candle.close_time + timedelta(seconds=offset - 1)
                noise_candle = build_synthetic_candle(
                    symbol="AVAXUSDT",
                    open_time=noise_time,
                    duration_seconds=1,
                    open_price=10.03,
                    high_price=10.05,
                    low_price=10.01,
                    close_price=10.04,
                )
                res = orchestrator.process_candle_event(noise_candle)
                assert (
                    res.signal.signal_type == SignalType.HOLD
                ), f"Failed at offset {offset}s within 300s reversal cooldown"
                assert engine.active_position is None

            unblock_time = rev_candle.close_time + timedelta(seconds=301)
            unblock_candle = build_synthetic_candle(
                symbol="AVAXUSDT",
                open_time=unblock_time,
                duration_seconds=60,
                open_price=10.00,
                high_price=10.02,
                low_price=9.98,
                close_price=10.00,
            )
            res_unblock = orchestrator.process_candle_event(unblock_candle)
            assert res_unblock.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None
        finally:
            ledger.close()


@pytest.mark.parametrize(
    "rsi_val,fast_ema,slow_ema,expected_signal",
    [
        (65.0, 100.5, 99.5, SignalType.HOLD),
        (58.5, 100.5, 99.5, SignalType.HOLD),
        (54.0, 100.5, 99.5, SignalType.HOLD),
        (45.0, 100.5, 99.5, SignalType.HOLD),
        (35.0, 100.5, 99.5, SignalType.HOLD),
        (32.1, 100.5, 99.5, SignalType.HOLD),
        (32.0001, 100.5, 99.5, SignalType.HOLD),
        (20.0, 99.0, 100.0, SignalType.HOLD),
        (15.0, 98.5, 101.0, SignalType.HOLD),
        (31.9, 99.5, 100.0, SignalType.HOLD),
        (32.0, 99.9999, 100.0, SignalType.HOLD),
        (32.0, 100.0, 100.0, SignalType.BUY),
        (32.0, 100.5, 100.0, SignalType.BUY),
        (25.0, 100.5, 100.0, SignalType.BUY),
    ],
)
def test_bearish_macro_regime_filtering_matrix(
    monkeypatch: pytest.MonkeyPatch,
    rsi_val: float,
    fast_ema: float,
    slow_ema: float,
    expected_signal: SignalType,
) -> None:
    """Stress-test BEARISH regime boundary conditions for breakouts and knife drops.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.
        rsi_val: Synthetic RSI value.
        fast_ema: Fast EMA indicator value.
        slow_ema: Slow EMA indicator value.
        expected_signal: Expected filtered SignalType.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_bearish_matrix.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)
            t0 = datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=100.0, count=35)

            eval_time = t0 + timedelta(minutes=35)
            synthetic_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=eval_time,
                signal_type=SignalType.BUY,
                reference_price=100.0,
                stop_loss_price=99.5,
                take_profit_price=102.0,
                atr_value=0.5,
                rsi_value=rsi_val,
                fast_ema=fast_ema,
                slow_ema=slow_ema,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: synthetic_signal,
            )
            test_candle = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=eval_time,
                duration_seconds=60,
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.0,
            )
            eval_result = orchestrator.process_candle_event(test_candle)
            assert eval_result.signal.signal_type == expected_signal
        finally:
            ledger.close()


def test_dynamic_regime_transition_suppresses_identical_breakout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify switching to BEARISH suppresses previously valid breakout.

    Args:
        monkeypatch: Pytest fixture for monkeypatching dependencies.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_regime_transition.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(symbol="SOLUSDT"),
                ledger_db_path=db_path,
            )
            engine = PaperTradingEngine(
                capital_config=config.capital,
                exchange_config=config.exchange,
                risk_config=config.risk,
                ledger=ledger,
            )
            mock_provider = MagicMock()
            orchestrator = RealTimeTradingOrchestrator(
                config=config,
                market_data_provider=mock_provider,
                trading_engine=engine,
            )
            t0 = datetime(2026, 11, 4, 19, 0, tzinfo=timezone.utc)
            bootstrap_orchestrator(orchestrator, t0, base_price=100.0, count=35)

            eval_time_1 = t0 + timedelta(minutes=35)
            breakout_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=eval_time_1,
                signal_type=SignalType.BUY,
                reference_price=100.0,
                stop_loss_price=99.5,
                take_profit_price=103.0,
                atr_value=0.5,
                rsi_value=53.0,
                fast_ema=100.5,
                slow_ema=99.5,
            )
            monkeypatch.setattr(
                "crypto_paper_trader.orchestration.evaluate_market_signal",
                lambda **kwargs: breakout_signal,
            )
            orchestrator.set_macro_regime(MacroMarketRegime.BULLISH)
            candle_1 = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=eval_time_1,
                duration_seconds=60,
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.0,
            )
            result_bullish = orchestrator.process_candle_event(candle_1)
            assert result_bullish.signal.signal_type == SignalType.BUY
            assert engine.active_position is not None

            engine._active_position = None
            eval_time_2 = t0 + timedelta(minutes=36)
            orchestrator.set_macro_regime(MacroMarketRegime.BEARISH)
            candle_2 = build_synthetic_candle(
                symbol="SOLUSDT",
                open_time=eval_time_2,
                duration_seconds=60,
                open_price=100.0,
                high_price=100.2,
                low_price=99.8,
                close_price=100.0,
            )
            result_bearish = orchestrator.process_candle_event(candle_2)
            assert result_bearish.signal.signal_type == SignalType.HOLD
            assert engine.active_position is None
        finally:
            ledger.close()


def test_trailing_stop_breathing_room_and_fee_immunity_under_volatile_spikes() -> None:
    """Stress-test trailing stop resilience against micro-noise dips.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_trailing_breathing.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(
                    symbol="SOLUSDT",
                    taker_fee_rate=0.0010,
                ),
                risk=RiskConfig(
                    min_net_profit_ratio=0.0010,
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
            t0 = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=100.00,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=48.0,
                fast_ema=100.10,
                slow_ema=99.90,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None
            assert engine.active_position is not None

            t1 = t0 + timedelta(minutes=1)
            engine.evaluate_protective_exits(current_price=100.24, timestamp=t1)
            pos = engine.active_position
            assert pos is not None
            assert pos.stop_loss_price == 100.22

            micro_noise_prices = [100.235, 100.228, 100.231, 100.225]
            for idx, noise_price in enumerate(micro_noise_prices):
                t_noise = t1 + timedelta(seconds=(idx + 1) * 10)
                exit_res = engine.evaluate_protective_exits(
                    current_price=noise_price, timestamp=t_noise
                )
                assert (
                    exit_res is None
                ), f"Premature stop-out triggered at micro-noise price {noise_price}"
                assert engine.active_position is not None

            t2 = t1 + timedelta(minutes=2)
            engine.evaluate_protective_exits(current_price=100.35, timestamp=t2)
            pos_t2 = engine.active_position
            assert pos_t2 is not None
            assert pos_t2.stop_loss_price >= 100.27

            t3 = t2 + timedelta(minutes=1)
            tier1_noise_prices = [100.32, 100.29, 100.31]
            for idx, noise_price in enumerate(tier1_noise_prices):
                t_tier1_noise = t3 + timedelta(seconds=(idx + 1) * 10)
                exit_res = engine.evaluate_protective_exits(
                    current_price=noise_price, timestamp=t_tier1_noise
                )
                assert exit_res is None
                assert engine.active_position is not None

            t4 = t3 + timedelta(minutes=2)
            final_exit = engine.evaluate_protective_exits(
                current_price=100.26, timestamp=t4
            )
            assert final_exit is not None
            assert final_exit.reason == ExitReason.TRAILING_STOP.value
            assert final_exit.realized_pnl_usdt > 0.0
            round_trip_fee_rate = config.exchange.taker_fee_rate * 2.0
            gross_pnl_ratio = (
                final_exit.execution_price - buy_exec.execution_price
            ) / buy_exec.execution_price
            assert gross_pnl_ratio > round_trip_fee_rate
            assert engine.active_position is None
        finally:
            ledger.close()


@pytest.mark.parametrize(
    "exit_price,should_execute",
    [
        (100.05, False),
        (100.10, False),
        (100.18, False),
        (100.22, False),
        (100.25, False),
        (100.280, False),
        (100.299, False),
        (100.300, True),
        (100.350, True),
    ],
)
def test_signal_reversal_fee_immunity_sweep(
    exit_price: float,
    should_execute: bool,
) -> None:
    """Verify signal reversal enforces positive net returns across a price sweep.

    Args:
        exit_price: Target exit price to evaluate.
        should_execute: Expected boolean execution status.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_rev_sweep.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(
                    symbol="SOLUSDT",
                    taker_fee_rate=0.0010,
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
            t0 = datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=100.00,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=48.0,
                fast_ema=100.10,
                slow_ema=99.90,
            )
            buy_exec = engine.process_signal(buy_signal)
            assert buy_exec is not None

            sell_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0 + timedelta(minutes=5),
                signal_type=SignalType.SELL,
                reference_price=exit_price,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=60.0,
                fast_ema=100.20,
                slow_ema=99.90,
            )
            sell_exec = engine.process_signal(sell_signal)
            if should_execute:
                assert sell_exec is not None
                assert sell_exec.reason == ExitReason.SIGNAL_REVERSAL.value
                assert sell_exec.realized_pnl_usdt > 0.0
                assert engine.active_position is None
            else:
                assert sell_exec is None
                assert engine.active_position is not None
        finally:
            ledger.close()


def test_dynamic_fee_adaptation_with_elevated_exchange_fees() -> None:
    """Verify execution engine dynamically adjusts reversal exit when fees increase.

    Returns:
        None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_elevated_fees.db"
        ledger = LedgerRepository(db_path=db_path)
        try:
            config = TradingSystemConfig(
                exchange=ExchangeConfig(
                    symbol="SOLUSDT",
                    taker_fee_rate=0.0015,
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
            t0 = datetime(2026, 11, 4, 22, 0, tzinfo=timezone.utc)
            buy_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0,
                signal_type=SignalType.BUY,
                reference_price=100.00,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=48.0,
                fast_ema=100.10,
                slow_ema=99.90,
            )
            engine.process_signal(buy_signal)

            sub_fee_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0 + timedelta(minutes=5),
                signal_type=SignalType.SELL,
                reference_price=100.35,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=60.0,
                fast_ema=100.20,
                slow_ema=99.90,
            )
            sub_res = engine.process_signal(sub_fee_signal)
            assert sub_res is None
            assert engine.active_position is not None

            immune_signal = MarketSignal(
                symbol="SOLUSDT",
                evaluated_at=t0 + timedelta(minutes=10),
                signal_type=SignalType.SELL,
                reference_price=100.41,
                stop_loss_price=99.50,
                take_profit_price=103.00,
                atr_value=0.50,
                rsi_value=60.0,
                fast_ema=100.20,
                slow_ema=99.90,
            )
            immune_res = engine.process_signal(immune_signal)
            assert immune_res is not None
            assert immune_res.reason == ExitReason.SIGNAL_REVERSAL.value
            assert immune_res.realized_pnl_usdt > 0.0
            assert engine.active_position is None
        finally:
            ledger.close()
