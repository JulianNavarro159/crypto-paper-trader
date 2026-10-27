from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from crypto_paper_trader.config import CapitalConfig, ExchangeConfig, RiskConfig
from crypto_paper_trader.domain import (
    ExitReason,
    MarketSignal,
    OrderSide,
    PortfolioSnapshot,
    Position,
    SignalType,
    TradeExecution,
)
from crypto_paper_trader.live_exchange import BinanceSpotAuthenticatedClient
from crypto_paper_trader.notifications import (
    NotificationService,
    NullNotificationService,
    format_execution_alert,
)
from crypto_paper_trader.validation import (
    OrderValidationError,
    validate_entry_order_viability,
)

logger = logging.getLogger(__name__)


def generate_client_order_id(
    symbol: str,
    side: OrderSide,
    timestamp: datetime,
    reason: str,
) -> str:
    """Generate a deterministic idempotency key for an order execution.

    Args:
        symbol: Trading pair identifier.
        side: Order direction (BUY or SELL).
        timestamp: UTC timestamp of the triggering event.
        reason: Execution trigger reason.

    Returns:
        Hexadecimal digest string uniquely identifying the order.
    """
    raw_key = f"{symbol}:{side.value}:{timestamp.isoformat()}:{reason}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
    return f"paper-{symbol.lower()}-{side.value.lower()}-{digest}"


class LedgerRepository:
    """SQLite-backed audit repository for idempotent trade and equity persistence.

    Args:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Provide a cached thread-safe SQLite connection with WAL mode enabled.

        Returns:
            Active SQLite connection instance.
        """
        if self._connection is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._connection = conn
        return self._connection

    def close(self) -> None:
        """Close the underlying SQLite connection if open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> LedgerRepository:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        """Create database tables if they do not already exist."""
        connection = self._get_connection()
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_executions (
                client_order_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                executed_at TEXT NOT NULL,
                execution_price REAL NOT NULL,
                quantity REAL NOT NULL,
                gross_notional_usdt REAL NOT NULL,
                fee_usdt REAL NOT NULL,
                realized_pnl_usdt REAL NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                available_cash_usdt REAL NOT NULL,
                open_position_value_usdt REAL NOT NULL,
                total_equity_usdt REAL NOT NULL,
                total_equity_cop REAL NOT NULL,
                realized_pnl_usdt REAL NOT NULL,
                total_fees_paid_usdt REAL NOT NULL,
                total_trades INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS open_positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional_usdt REAL NOT NULL,
                entry_fee_usdt REAL NOT NULL,
                stop_loss_price REAL NOT NULL,
                take_profit_price REAL NOT NULL,
                peak_price REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        try:
            connection.execute(
                "ALTER TABLE open_positions "
                "ADD COLUMN peak_price REAL NOT NULL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass
        connection.commit()

    def save_open_position(self, position: Position) -> None:
        """Persist the currently active position so it survives process restarts.

        Args:
            position: Open Position domain model to store.
        """
        connection = self._get_connection()
        connection.execute("DELETE FROM open_positions")
        connection.execute(
            """
            INSERT INTO open_positions (
                position_id,
                symbol,
                opened_at,
                entry_price,
                quantity,
                notional_usdt,
                entry_fee_usdt,
                stop_loss_price,
                take_profit_price,
                peak_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.position_id,
                position.symbol,
                position.opened_at.isoformat(),
                position.entry_price,
                position.quantity,
                position.notional_usdt,
                position.entry_fee_usdt,
                position.stop_loss_price,
                position.take_profit_price,
                position.peak_price,
            ),
        )
        connection.commit()

    def clear_open_position(self) -> None:
        """Remove any persisted open position when a trade is closed."""
        connection = self._get_connection()
        connection.execute("DELETE FROM open_positions")
        connection.commit()

    def fetch_open_position(self) -> Position | None:
        """Retrieve a previously opened position if one remains active.

        Returns:
            Restored Position instance if an open trade exists, else None.
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            SELECT
                position_id,
                symbol,
                opened_at,
                entry_price,
                quantity,
                notional_usdt,
                entry_fee_usdt,
                stop_loss_price,
                take_profit_price,
                peak_price
            FROM open_positions
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return Position(
            position_id=str(row[0]),
            symbol=str(row[1]),
            opened_at=datetime.fromisoformat(row[2]),
            entry_price=float(row[3]),
            quantity=float(row[4]),
            notional_usdt=float(row[5]),
            entry_fee_usdt=float(row[6]),
            stop_loss_price=float(row[7]),
            take_profit_price=float(row[8]),
            peak_price=float(row[9]),
        )

    def record_execution(self, execution: TradeExecution) -> bool:
        """Persist a trade execution idempotently using client_order_id.

        Args:
            execution: TradeExecution domain instance to persist.

        Returns:
            True if a new row was inserted, False if client_order_id already existed.
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO trade_executions (
                client_order_id,
                position_id,
                symbol,
                side,
                executed_at,
                execution_price,
                quantity,
                gross_notional_usdt,
                fee_usdt,
                realized_pnl_usdt,
                reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution.client_order_id,
                execution.position_id,
                execution.symbol,
                execution.side.value,
                execution.executed_at.isoformat(),
                execution.execution_price,
                execution.quantity,
                execution.gross_notional_usdt,
                execution.fee_usdt,
                execution.realized_pnl_usdt,
                execution.reason,
            ),
        )
        connection.commit()
        return cursor.rowcount > 0

    def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Persist a portfolio valuation snapshot to the audit table.

        Args:
            snapshot: PortfolioSnapshot domain instance to store.
        """
        connection = self._get_connection()
        connection.execute(
            """
            INSERT INTO portfolio_snapshots (
                timestamp,
                available_cash_usdt,
                open_position_value_usdt,
                total_equity_usdt,
                total_equity_cop,
                realized_pnl_usdt,
                total_fees_paid_usdt,
                total_trades
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.timestamp.isoformat(),
                snapshot.available_cash_usdt,
                snapshot.open_position_value_usdt,
                snapshot.total_equity_usdt,
                snapshot.total_equity_cop,
                snapshot.realized_pnl_usdt,
                snapshot.total_fees_paid_usdt,
                snapshot.total_trades,
            ),
        )
        connection.commit()

    def fetch_all_executions(self) -> list[TradeExecution]:
        """Retrieve all stored trade executions ordered chronologically.

        Returns:
            List of persisted TradeExecution records.
        """
        connection = self._get_connection()
        cursor = connection.execute(
            """
            SELECT
                client_order_id,
                position_id,
                symbol,
                side,
                executed_at,
                execution_price,
                quantity,
                gross_notional_usdt,
                fee_usdt,
                realized_pnl_usdt,
                reason
            FROM trade_executions
            ORDER BY executed_at ASC
            """
        )
        rows = cursor.fetchall()

        return [
            TradeExecution(
                client_order_id=row[0],
                position_id=row[1],
                symbol=row[2],
                side=OrderSide(row[3]),
                executed_at=datetime.fromisoformat(row[4]),
                execution_price=float(row[5]),
                quantity=float(row[6]),
                gross_notional_usdt=float(row[7]),
                fee_usdt=float(row[8]),
                realized_pnl_usdt=float(row[9]),
                reason=str(row[10]),
            )
            for row in rows
        ]


class PaperTradingEngine:
    """Stateful, idempotent real-time paper trading execution engine.

    Args:
        capital_config: Starting capital and currency exchange rate settings.
        exchange_config: Exchange commission and minimum notional settings.
        risk_config: Position sizing and fee-aware profit constraints.
        ledger: SQLite ledger repository for persisting executions and snapshots.
    """

    def __init__(
        self,
        capital_config: CapitalConfig,
        exchange_config: ExchangeConfig,
        risk_config: RiskConfig,
        ledger: LedgerRepository,
        notifier: NotificationService | None = None,
        live_client: BinanceSpotAuthenticatedClient | None = None,
    ) -> None:
        self._capital_config = capital_config
        self._exchange_config = exchange_config
        self._risk_config = risk_config
        self._ledger = ledger
        self._notifier = notifier if notifier is not None else NullNotificationService()
        self._live_client = live_client

        if self._live_client is not None:
            try:
                self._available_cash_usdt = (
                    self._live_client.fetch_available_usdt_balance()
                )
            except Exception as exc:
                fallback_cash = capital_config.initial_capital_usdt
                logger.warning(
                    "Could not refresh live Binance USDT balance (%s); "
                    "using fallback %.4f USDT",
                    exc,
                    fallback_cash,
                )
                self._available_cash_usdt = fallback_cash
        else:
            self._available_cash_usdt = capital_config.initial_capital_usdt
        self._active_position: Position | None = self._ledger.fetch_open_position()
        if self._active_position is not None:
            if self._live_client is None:
                self._available_cash_usdt = round(
                    max(
                        0.0,
                        self._available_cash_usdt - self._active_position.notional_usdt,
                    ),
                    6,
                )
            logger.info(
                "Resumed open position from ledger | symbol=%s entry=%.4f "
                "qty=%.6f sl=%.4f tp=%.4f",
                self._active_position.symbol,
                self._active_position.entry_price,
                self._active_position.quantity,
                self._active_position.stop_loss_price,
                self._active_position.take_profit_price,
            )
        self._realized_pnl_usdt: float = 0.0
        self._total_fees_paid_usdt: float = 0.0
        self._total_trades: int = 0
        self._executed_order_ids: set[str] = set()

    @property
    def available_cash_usdt(self) -> float:
        """Return current uninvested USDT balance."""
        return self._available_cash_usdt

    @property
    def active_position(self) -> Position | None:
        """Return currently open paper position or None."""
        return self._active_position

    def evaluate_protective_exits(
        self,
        current_price: float,
        timestamp: datetime,
    ) -> TradeExecution | None:
        """Check whether an open position has triggered stop loss or take profit.

        Args:
            current_price: Latest market trade or close price.
            timestamp: UTC timestamp of the price observation.

        Returns:
            TradeExecution if the position was closed, otherwise None.
        """
        if self._active_position is None:
            return None

        pos = self._active_position
        new_peak = max(pos.peak_price, current_price, pos.entry_price)
        gain_ratio = (new_peak - pos.entry_price) / pos.entry_price

        if gain_ratio >= 0.012:
            breakeven_floor = round(pos.entry_price * 1.004, 8)
            trailing_distance = round(pos.entry_price * 0.010, 8)
            trailing_candidate = round(new_peak - trailing_distance, 8)
            new_sl = max(pos.stop_loss_price, breakeven_floor, trailing_candidate)
            if new_sl > pos.stop_loss_price or new_peak > pos.peak_price:
                self._active_position = replace(
                    pos, stop_loss_price=new_sl, peak_price=new_peak
                )
                self._ledger.save_open_position(self._active_position)
        elif new_peak > pos.peak_price:
            self._active_position = replace(pos, peak_price=new_peak)
            self._ledger.save_open_position(self._active_position)

        if current_price <= self._active_position.stop_loss_price:
            reason = (
                ExitReason.TRAILING_STOP.value
                if self._active_position.stop_loss_price
                >= self._active_position.entry_price
                else ExitReason.STOP_LOSS.value
            )
            return self._close_active_position(
                exit_price=current_price,
                timestamp=timestamp,
                reason=reason,
            )

        if current_price >= self._active_position.take_profit_price:
            return self._close_active_position(
                exit_price=current_price,
                timestamp=timestamp,
                reason=ExitReason.TAKE_PROFIT.value,
            )

        return None

    def process_signal(self, signal: MarketSignal) -> TradeExecution | None:
        """Process a strategy MarketSignal and execute paper orders if valid.

        Args:
            signal: Evaluated MarketSignal from the technical strategy layer.

        Returns:
            TradeExecution if an entry or exit order was filled, otherwise None.
        """
        if self._active_position is not None and signal.signal_type == SignalType.SELL:
            min_profitable_exit = self._active_position.entry_price * (
                1.0 + self._risk_config.min_signal_exit_profit_ratio
            )
            if signal.reference_price >= min_profitable_exit:
                return self._close_active_position(
                    exit_price=signal.reference_price,
                    timestamp=signal.evaluated_at,
                    reason=ExitReason.SIGNAL_REVERSAL.value,
                )
            return None

        if self._active_position is None and signal.signal_type == SignalType.BUY:
            return self._open_new_position(signal)

        return None

    def _open_new_position(self, signal: MarketSignal) -> TradeExecution | None:
        """Validate and execute a simulated BUY entry order.

        Args:
            signal: BUY MarketSignal with entry, stop loss, and take profit prices.

        Returns:
            TradeExecution if the order passed validation and filled, else None.
        """
        client_order_id = generate_client_order_id(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            timestamp=signal.evaluated_at,
            reason="SIGNAL_ENTRY",
        )
        if client_order_id in self._executed_order_ids:
            return None

        proposed_notional_usdt = round(
            self._available_cash_usdt * self._risk_config.position_allocation_fraction,
            4,
        )

        try:
            validate_entry_order_viability(
                signal=signal,
                proposed_notional_usdt=proposed_notional_usdt,
                available_cash_usdt=self._available_cash_usdt,
                exchange_config=self._exchange_config,
                risk_config=self._risk_config,
            )
        except OrderValidationError as exc:
            logger.warning("Skipped BUY signal due to risk validation: %s", exc)
            return None

        if self._live_client is not None:
            self._live_client.place_market_buy_by_quote(
                symbol=signal.symbol,
                quote_amount_usdt=proposed_notional_usdt,
                client_order_id=client_order_id,
            )

        entry_fee_usdt = round(
            proposed_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        net_notional_usdt = proposed_notional_usdt - entry_fee_usdt
        quantity = round(net_notional_usdt / signal.reference_price, 8)
        position_id = f"pos-{client_order_id[-12:]}"

        self._available_cash_usdt = round(
            self._available_cash_usdt - proposed_notional_usdt, 6
        )
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + entry_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)

        self._active_position = Position(
            position_id=position_id,
            symbol=signal.symbol,
            opened_at=signal.evaluated_at,
            entry_price=signal.reference_price,
            quantity=quantity,
            notional_usdt=proposed_notional_usdt,
            entry_fee_usdt=entry_fee_usdt,
            stop_loss_price=signal.stop_loss_price,
            take_profit_price=signal.take_profit_price,
            peak_price=signal.reference_price,
        )
        self._ledger.save_open_position(self._active_position)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=position_id,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            executed_at=signal.evaluated_at,
            execution_price=signal.reference_price,
            quantity=quantity,
            gross_notional_usdt=proposed_notional_usdt,
            fee_usdt=entry_fee_usdt,
            realized_pnl_usdt=0.0,
            reason="SIGNAL_ENTRY",
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        logger.info(
            "Executed %s BUY | symbol=%s price=%.4f qty=%.6f notional=%.4f USDT "
            "sl=%.4f tp=%.4f",
            mode_tag,
            execution.symbol,
            execution.execution_price,
            execution.quantity,
            execution.gross_notional_usdt,
            signal.stop_loss_price,
            signal.take_profit_price,
        )
        self._notifier.send_alert(
            format_execution_alert(
                execution=execution,
                available_cash_usdt=self._available_cash_usdt,
                cop_per_usdt_rate=self._capital_config.cop_per_usdt_rate,
                is_live_order=self._live_client is not None,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
            )
        )
        return execution

    def _close_active_position(
        self,
        exit_price: float,
        timestamp: datetime,
        reason: str,
    ) -> TradeExecution | None:
        """Close the currently open position and realize net profit or loss.

        Args:
            exit_price: Market fill price per unit of base asset.
            timestamp: UTC timestamp of exit execution.
            reason: Exit trigger classifier string.

        Returns:
            TradeExecution representing the closing SELL order.
        """
        if self._active_position is None:
            return None

        position = self._active_position
        client_order_id = generate_client_order_id(
            symbol=position.symbol,
            side=OrderSide.SELL,
            timestamp=timestamp,
            reason=reason,
        )
        if client_order_id in self._executed_order_ids:
            return None

        if self._live_client is not None:
            self._live_client.place_market_sell_by_quantity(
                symbol=position.symbol,
                quantity=position.quantity,
                client_order_id=client_order_id,
            )

        gross_exit_notional_usdt = round(position.quantity * exit_price, 6)
        exit_fee_usdt = round(
            gross_exit_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        net_exit_proceeds_usdt = gross_exit_notional_usdt - exit_fee_usdt
        realized_pnl_usdt = round(net_exit_proceeds_usdt - position.notional_usdt, 6)

        self._available_cash_usdt = round(
            self._available_cash_usdt + net_exit_proceeds_usdt, 6
        )
        self._realized_pnl_usdt = round(self._realized_pnl_usdt + realized_pnl_usdt, 6)
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + exit_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)
        self._active_position = None
        self._ledger.clear_open_position()

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=position.position_id,
            symbol=position.symbol,
            side=OrderSide.SELL,
            executed_at=timestamp,
            execution_price=exit_price,
            quantity=position.quantity,
            gross_notional_usdt=gross_exit_notional_usdt,
            fee_usdt=exit_fee_usdt,
            realized_pnl_usdt=realized_pnl_usdt,
            reason=reason,
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        logger.info(
            "Executed %s SELL | symbol=%s reason=%s price=%.4f "
            "net_pnl=%.4f USDT cash=%.4f USDT",
            mode_tag,
            execution.symbol,
            execution.reason,
            execution.execution_price,
            execution.realized_pnl_usdt,
            self._available_cash_usdt,
        )
        self._notifier.send_alert(
            format_execution_alert(
                execution=execution,
                available_cash_usdt=self._available_cash_usdt,
                cop_per_usdt_rate=self._capital_config.cop_per_usdt_rate,
                is_live_order=self._live_client is not None,
            )
        )
        return execution

    def capture_portfolio_snapshot(
        self,
        current_price: float,
        timestamp: datetime,
    ) -> PortfolioSnapshot:
        """Calculate and persist a mark-to-market portfolio valuation snapshot.

        Args:
            current_price: Latest market price used to value open holdings.
            timestamp: UTC timestamp of the valuation.

        Returns:
            Persisted PortfolioSnapshot domain record.
        """
        open_position_value_usdt = 0.0
        if self._active_position is not None:
            gross_value = self._active_position.quantity * current_price
            estimated_exit_fee = gross_value * self._exchange_config.taker_fee_rate
            open_position_value_usdt = round(gross_value - estimated_exit_fee, 6)

        total_equity_usdt = round(
            self._available_cash_usdt + open_position_value_usdt, 4
        )
        total_equity_cop = round(
            total_equity_usdt * self._capital_config.cop_per_usdt_rate, 2
        )

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            available_cash_usdt=round(self._available_cash_usdt, 4),
            open_position_value_usdt=round(open_position_value_usdt, 4),
            total_equity_usdt=total_equity_usdt,
            total_equity_cop=total_equity_cop,
            realized_pnl_usdt=round(self._realized_pnl_usdt, 4),
            total_fees_paid_usdt=round(self._total_fees_paid_usdt, 4),
            total_trades=self._total_trades,
        )
        self._ledger.record_snapshot(snapshot)
        return snapshot
