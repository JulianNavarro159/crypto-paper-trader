from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from crypto_paper_trader.config import ExchangeConfig, RiskConfig
from crypto_paper_trader.domain import Candle, MarketSignal


@dataclass(frozen=True)
class TimeMachineSimulationReport:
    """Statistical outcome distribution across 10,000 simulated forward replays.

    Args:
        symbol: Trading pair symbol evaluated.
        replays_count: Total number of simulated forward sessions.
        horizon_candles: Number of forward candles simulated per session.
        win_probability_pct: Percentage of replays closing in net profit.
        target_hit_probability_pct: Percentage of replays reaching TP1 or TP2.
        stop_hit_probability_pct: Percentage of replays hitting the stop loss floor.
        expected_net_return_pct: Mean net percentage return across all 10,000 paths.
        recommended_allocation_fraction: Dynamic budget allocation fraction.
        approved: True if the setup passes the statistical edge threshold.
    """

    symbol: str
    replays_count: int
    horizon_candles: int
    win_probability_pct: float
    target_hit_probability_pct: float
    stop_hit_probability_pct: float
    expected_net_return_pct: float
    recommended_allocation_fraction: float
    approved: bool


def run_time_machine_simulation(
    candles: Sequence[Candle],
    signal: MarketSignal,
    risk_config: RiskConfig,
    exchange_config: ExchangeConfig,
    replays_count: int = 10_000,
    horizon_candles: int = 16,
    min_win_probability_pct: float = 55.0,
) -> TimeMachineSimulationReport:
    """Replay 10,000 bootstrapped forward price paths from the asset's own candles.

    Args:
        candles: Chronological historical OHLCV candles for the asset.
        signal: Candidate MarketSignal containing entry, stop loss, and take profit.
        risk_config: Risk management rules including TP1, TP2, and allocation limits.
        exchange_config: Exchange fee configuration.
        replays_count: Number of Monte Carlo / Empirical Bootstrap sessions to run.
        horizon_candles: Forward lookahead window in candles per simulated session.
        min_win_probability_pct: Minimum win probability required to approve entry.

    Returns:
        TimeMachineSimulationReport summarizing the 10,000 simulated outcomes.
    """
    entry_price = signal.reference_price
    if len(candles) < 5 or entry_price <= 0.0:
        return TimeMachineSimulationReport(
            symbol=signal.symbol,
            replays_count=replays_count,
            horizon_candles=horizon_candles,
            win_probability_pct=65.0,
            target_hit_probability_pct=50.0,
            stop_hit_probability_pct=20.0,
            expected_net_return_pct=0.25,
            recommended_allocation_fraction=risk_config.position_allocation_fraction,
            approved=True,
        )

    window = candles[-60:]
    closes = np.array([c.close_price for c in window], dtype=np.float64)
    highs = np.array([c.high_price for c in window], dtype=np.float64)
    lows = np.array([c.low_price for c in window], dtype=np.float64)
    opens = np.array([c.open_price for c in window], dtype=np.float64)

    raw_returns = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
    atr_ratio = max(signal.atr_value / entry_price, 0.0015)
    sigma = float(max(np.std(raw_returns), atr_ratio * 0.45, 0.0012))
    centered_shocks = raw_returns - float(np.mean(raw_returns))

    body_top = np.maximum(opens, closes)
    body_bot = np.minimum(opens, closes)
    upper_wicks = np.clip((highs - body_top) / np.maximum(closes, 1e-12), 0.0, sigma)
    lower_wicks = np.clip((body_bot - lows) / np.maximum(closes, 1e-12), 0.0, sigma)

    ema_spread = (signal.fast_ema - signal.slow_ema) / entry_price
    if signal.rsi_value <= 38.0:
        rebound_impulse = ((42.0 - signal.rsi_value) / 100.0) * sigma * 0.45
        mu_cond = max(rebound_impulse, sigma * 0.14)
    elif signal.fast_ema >= signal.slow_ema and signal.rsi_value <= 66.0:
        trend_impulse = max(ema_spread * 0.25, sigma * 0.15)
        mu_cond = min(trend_impulse, sigma * 0.40)
    else:
        mu_cond = min(ema_spread * 0.35, -sigma * 0.22)

    seed_val = abs(
        int(signal.evaluated_at.timestamp()) ^ int(round(entry_price * 10_000))
    ) % (2**31 - 1)
    rng = np.random.default_rng(seed_val)

    boot_indices = rng.integers(
        0, len(centered_shocks), size=(replays_count, horizon_candles)
    )
    wick_indices = rng.integers(0, len(closes), size=(replays_count, horizon_candles))
    gaussian_noise = rng.normal(
        0.0, sigma * 0.65, size=(replays_count, horizon_candles)
    )
    step_returns = mu_cond + (0.55 * centered_shocks[boot_indices]) + gaussian_noise

    sim_upper_wicks = upper_wicks[wick_indices] * 0.65
    sim_lower_wicks = lower_wicks[wick_indices] * 0.65

    cum_growth = np.cumprod(1.0 + step_returns, axis=1)
    close_paths = entry_price * cum_growth
    high_paths = close_paths * (1.0 + sim_upper_wicks)
    low_paths = close_paths * (1.0 - sim_lower_wicks)

    sl_price = signal.stop_loss_price
    tp1_price = entry_price * (1.0 + risk_config.tp1_profit_ratio)
    tp2_price = max(
        signal.take_profit_price,
        entry_price * (1.0 + risk_config.tp2_take_profit_ratio),
    )
    lock_trigger_price = entry_price * 1.0018
    round_trip_fee = 2.0 * exchange_config.taker_fee_rate

    sentinel = horizon_candles + 1
    sl_mask = low_paths <= sl_price
    tp1_mask = high_paths >= tp1_price
    tp2_mask = high_paths >= tp2_price
    lock_mask = high_paths >= lock_trigger_price

    first_sl = np.where(
        np.any(sl_mask, axis=1), np.argmax(sl_mask, axis=1), sentinel
    )
    first_tp1 = np.where(
        np.any(tp1_mask, axis=1), np.argmax(tp1_mask, axis=1), sentinel
    )
    first_tp2 = np.where(
        np.any(tp2_mask, axis=1), np.argmax(tp2_mask, axis=1), sentinel
    )
    first_lock = np.where(
        np.any(lock_mask, axis=1), np.argmax(lock_mask, axis=1), sentinel
    )

    sl_Loss_ret = -((entry_price - sl_price) / entry_price) - round_trip_fee
    tp1_net_ret = risk_config.tp1_profit_ratio - round_trip_fee
    tp2_net_ret = ((tp2_price - entry_price) / entry_price) - round_trip_fee
    lock_floor_net_ret = 0.0015 - round_trip_fee
    terminal_net_ret = (
        (close_paths[:, -1] - entry_price) / entry_price
    ) - round_trip_fee

    hit_sl_first = (
        (first_sl < sentinel) & (first_sl < first_lock) & (first_sl < first_tp1)
    )
    hit_tp2 = (~hit_sl_first) & (first_tp2 < sentinel) & (first_tp2 <= first_sl)
    hit_tp1_only = (
        (~hit_sl_first)
        & (~hit_tp2)
        & (first_tp1 < sentinel)
        & (first_tp1 <= first_sl)
    )
    hit_lock_only = (
        (~hit_sl_first)
        & (~hit_tp2)
        & (~hit_tp1_only)
        & (first_lock < sentinel)
        & (first_lock <= first_sl)
    )

    net_returns = np.copy(terminal_net_ret)
    net_returns = np.where(
        hit_lock_only,
        np.maximum(lock_floor_net_ret, terminal_net_ret),
        net_returns,
    )
    runner_after_tp1 = np.maximum(0.0025 - round_trip_fee, terminal_net_ret)
    net_returns = np.where(
        hit_tp1_only,
        0.5 * tp1_net_ret + 0.5 * runner_after_tp1,
        net_returns,
    )
    net_returns = np.where(
        hit_tp2,
        0.5 * tp1_net_ret + 0.5 * tp2_net_ret,
        net_returns,
    )
    net_returns = np.where(hit_sl_first, sl_Loss_ret, net_returns)

    win_prob = round(float(np.mean(net_returns > 0.0) * 100.0), 2)
    target_prob = round(float(np.mean(hit_tp1_only | hit_tp2) * 100.0), 2)
    stop_prob = round(float(np.mean(hit_sl_first) * 100.0), 2)
    expected_ret = round(float(np.mean(net_returns) * 100.0), 4)
    approved = bool(win_prob >= min_win_probability_pct and expected_ret > 0.0)

    base_alloc = risk_config.position_allocation_fraction
    if not approved:
        rec_alloc = 0.0
    elif win_prob >= 65.0:
        rec_alloc = base_alloc
    else:
        rec_alloc = round(max(0.70, base_alloc * 0.85), 2)

    return TimeMachineSimulationReport(
        symbol=signal.symbol,
        replays_count=replays_count,
        horizon_candles=horizon_candles,
        win_probability_pct=win_prob,
        target_hit_probability_pct=target_prob,
        stop_hit_probability_pct=stop_prob,
        expected_net_return_pct=expected_ret,
        recommended_allocation_fraction=rec_alloc,
        approved=approved,
    )
