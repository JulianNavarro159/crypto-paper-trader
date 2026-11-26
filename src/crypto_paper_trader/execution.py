from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
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
    SlotRole,
    TradeExecution,
)
from crypto_paper_trader.formatting import format_price, format_quantity
from crypto_paper_trader.live_exchange import (
    BinanceSpotAuthenticatedClient,
    quantize_step_size,
)
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


def compute_no_loss_exit_levels(
    entry_price: float,
    quantity: float,
    notional_usdt: float,
    taker_fee_rate: float,
    min_target_net_profit_usdt: float,
    include_live_spread_cushion: bool = False,
    min_floor_net_profit_usdt: float | None = None,
) -> tuple[float, float]:
    """Compute fee-covered safety floor and target profit exit price.

    Args:
        entry_price: Executed purchase price per unit of base asset.
        quantity: Base asset quantity held in the position.
        notional_usdt: Total USDT cost basis allocated to the position.
        taker_fee_rate: Fractional exchange fee rate per side.
        min_target_net_profit_usdt: Target net USDT gain required for take profit.
        include_live_spread_cushion: Add bid-ask spread cushion for live fills.
        min_floor_net_profit_usdt: Optional separate net USDT gain for the safety
            floor; defaults to min_target_net_profit_usdt when omitted.

    Returns:
        Tuple of (safety_floor_price, target_take_profit_price) strictly
        greater than or equal to entry_price.
    """
    safe_qty = max(quantity, 1e-12)
    safe_net_factor = max(1.0 - taker_fee_rate, 1e-9)
    implied_entry_cost = (safe_qty * entry_price) / safe_net_factor
    effective_cost_usdt = max(notional_usdt, implied_entry_cost)
    net_denom = safe_qty * safe_net_factor

    floor_profit_usdt = max(
        0.0,
        min_floor_net_profit_usdt
        if min_floor_net_profit_usdt is not None
        else min_target_net_profit_usdt,
    )
    target_profit_usdt = max(floor_profit_usdt, min_target_net_profit_usdt)

    floor_guard_usdt = 0.0010 if floor_profit_usdt <= 0.01 else 0.0015
    raw_floor_price = (
        effective_cost_usdt + floor_profit_usdt + floor_guard_usdt
    ) / net_denom
    safety_floor_price = round(max(entry_price, raw_floor_price), 10)

    target_guard_usdt = 0.0010 if target_profit_usdt <= 0.01 else 0.0015
    raw_target_price = (
        effective_cost_usdt + target_profit_usdt + target_guard_usdt
    ) / net_denom
    base_target_price = max(safety_floor_price, raw_target_price)

    if include_live_spread_cushion:
        cushion_ratio = 0.00025 if target_profit_usdt <= 0.01 else 0.00035
    else:
        cushion_ratio = 0.00005
    target_take_profit_price = round(
        max(safety_floor_price, base_target_price * (1.0 + cushion_ratio)),
        10,
    )
    return safety_floor_price, target_take_profit_price


def generate_client_order_id(
    symbol: str,
    side: OrderSide,
    timestamp: datetime,
    reason: str,
    slot_role: SlotRole | None = None,
) -> str:
    """Generate a deterministic idempotency key for an order execution.

    Args:
        symbol: Trading pair identifier.
        side: Order direction (BUY or SELL).
        timestamp: UTC timestamp of the triggering event.
        reason: Execution trigger reason.
        slot_role: Optional SCALP or SWING role prefix embedded for auditing.

    Returns:
        Deterministic identifier string uniquely identifying the order.
    """
    raw_key = f"{symbol}:{side.value}:{timestamp.isoformat()}:{reason}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
    if slot_role is not None:
        return (
            f"{slot_role.value.lower()}-{symbol.lower()}-"
            f"{side.value.lower()}-{digest[:14]}"
        )
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
                peak_price REAL NOT NULL DEFAULT 0.0,
                is_partially_closed INTEGER NOT NULL DEFAULT 0,
                slot_role TEXT NOT NULL DEFAULT 'SCALP',
                target_net_profit_usdt REAL NOT NULL DEFAULT 0.0,
                trough_price REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        migration_statements = (
            "ALTER TABLE open_positions "
            "ADD COLUMN peak_price REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE open_positions "
            "ADD COLUMN is_partially_closed INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE open_positions "
            "ADD COLUMN slot_role TEXT NOT NULL DEFAULT 'SCALP'",
            "ALTER TABLE open_positions "
            "ADD COLUMN target_net_profit_usdt REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE open_positions "
            "ADD COLUMN trough_price REAL NOT NULL DEFAULT 0.0",
        )
        for statement in migration_statements:
            try:
                connection.execute(statement)
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
                peak_price,
                is_partially_closed,
                slot_role,
                target_net_profit_usdt,
                trough_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if position.is_partially_closed else 0,
                position.slot_role.value,
                position.target_net_profit_usdt,
                position.trough_price,
            ),
        )
        connection.commit()

    def clear_open_position(self) -> None:
        """Remove any persisted open position when a trade is closed."""
        connection = self._get_connection()
        connection.execute("DELETE FROM open_positions")
        connection.commit()

    @staticmethod
    def _row_to_position(row: tuple[Any, ...]) -> Position:
        """Deserialize a SQLite open_positions row into a Position domain object."""
        is_partially_closed = bool(row[10]) if len(row) > 10 else False
        raw_role = str(row[11]).upper() if len(row) > 11 and row[11] else "SCALP"
        try:
            slot_role = SlotRole(raw_role)
        except ValueError:
            slot_role = SlotRole.SCALP
        target_profit = float(row[12]) if len(row) > 12 else 0.0
        trough_price = float(row[13]) if len(row) > 13 else 0.0
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
            is_partially_closed=is_partially_closed,
            slot_role=slot_role,
            target_net_profit_usdt=target_profit,
            trough_price=trough_price,
        )

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
                peak_price,
                is_partially_closed,
                slot_role,
                target_net_profit_usdt,
                trough_price
            FROM open_positions
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_position(row)

    def save_open_position_for_symbol(self, position: Position) -> None:
        """Persist or replace an open position scoped by symbol.

        Args:
            position: Open Position domain model to store.
        """
        connection = self._get_connection()
        connection.execute(
            "DELETE FROM open_positions WHERE UPPER(symbol) = UPPER(?)",
            (position.symbol,),
        )
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
                peak_price,
                is_partially_closed,
                slot_role,
                target_net_profit_usdt,
                trough_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if position.is_partially_closed else 0,
                position.slot_role.value,
                position.target_net_profit_usdt,
                position.trough_price,
            ),
        )
        connection.commit()

    def clear_open_position_for_symbol(self, symbol: str) -> None:
        """Remove a persisted open position for a specific trading symbol.

        Args:
            symbol: Trading pair symbol whose position has closed.
        """
        connection = self._get_connection()
        connection.execute(
            "DELETE FROM open_positions WHERE UPPER(symbol) = UPPER(?)",
            (symbol,),
        )
        connection.commit()

    def fetch_all_open_positions(self) -> list[Position]:
        """Retrieve all persisted active open positions sorted by open time.

        Returns:
            Chronological list of currently open Position records.
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
                peak_price,
                is_partially_closed,
                slot_role,
                target_net_profit_usdt,
                trough_price
            FROM open_positions
            ORDER BY opened_at ASC
            """
        )
        rows = cursor.fetchall()
        return [self._row_to_position(row) for row in rows]

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
        past_execs = self._ledger.fetch_all_executions()
        self._realized_pnl_usdt: float = round(
            sum(
                e.realized_pnl_usdt
                for e in past_execs
                if e.side == OrderSide.SELL
            ),
            6,
        )
        self._total_fees_paid_usdt: float = round(
            sum(e.fee_usdt for e in past_execs), 6
        )
        self._total_trades: int = len(past_execs)
        self._executed_order_ids: set[str] = {
            e.client_order_id for e in past_execs
        }
        self._dca_applied_position_ids: set[str] = {
            e.position_id for e in past_execs if e.reason == "SMART_DCA"
        }
        self._buy_blocked_symbols: set[str] = set()
        self._swing_candidate_symbols: set[str] = set()
        loaded_positions = self._ledger.fetch_all_open_positions()
        self._open_positions: dict[str, Position] = {
            pos.symbol.upper(): pos for pos in loaded_positions
        }
        self._active_position: Position | None = None
        self._sync_primary_active_position()
        known_live_usdt = (
            self._live_client.get_known_usdt_balance()
            if self._live_client is not None
            else None
        )
        if known_live_usdt is not None:
            self._available_cash_usdt = known_live_usdt
        else:
            open_notional_total = sum(
                pos.notional_usdt for pos in self._open_positions.values()
            )
            self._available_cash_usdt = round(
                max(
                    0.0,
                    capital_config.initial_capital_usdt
                    + self._realized_pnl_usdt
                    - open_notional_total,
                ),
                6,
            )
        for sym_key, pos in list(self._open_positions.items()):
            if self._risk_config.no_loss_spot_mode:
                upgraded_sl, upgraded_tp = self._compute_position_exit_levels(pos)
                target_profit = (
                    pos.target_net_profit_usdt
                    if pos.target_net_profit_usdt > 0.0
                    else (
                        self._risk_config.swing_target_net_profit_usdt
                        if pos.slot_role == SlotRole.SWING
                        else self._risk_config.min_target_net_profit_usdt
                    )
                )
                needs_update = (
                    pos.stop_loss_price < upgraded_sl
                    or (
                        pos.slot_role == SlotRole.SWING
                        and pos.take_profit_price < upgraded_tp
                    )
                    or pos.target_net_profit_usdt <= 0.0
                )
                if needs_update:
                    pos = replace(
                        pos,
                        stop_loss_price=max(pos.stop_loss_price, upgraded_sl),
                        take_profit_price=max(pos.take_profit_price, upgraded_tp)
                        if pos.slot_role == SlotRole.SWING
                        else upgraded_tp,
                        target_net_profit_usdt=target_profit,
                    )
                    self._open_positions[sym_key] = pos
                    self._ledger.save_open_position_for_symbol(pos)
            logger.info(
                "Resumed open position from ledger | symbol=%s role=%s entry=%s "
                "qty=%s sl=%s tp=%s",
                pos.symbol,
                pos.slot_role.value,
                format_price(pos.entry_price),
                format_quantity(pos.quantity),
                format_price(pos.stop_loss_price),
                format_price(pos.take_profit_price),
            )
        self._sync_primary_active_position()

    def _sync_primary_active_position(self) -> None:
        """Keep _active_position aligned with the primary open position."""
        self._active_position = next(iter(self._open_positions.values()), None)

    def _reconcile_external_clear(self) -> None:
        """Synchronize internal position map if ledger open positions were cleared."""
        if self._active_position is None and self._open_positions:
            if not self._ledger.fetch_all_open_positions():
                self._open_positions.clear()

    @property
    def no_loss_spot_mode(self) -> bool:
        """Return True when below-entry sales are strictly prohibited."""
        return self._risk_config.no_loss_spot_mode

    @property
    def available_cash_usdt(self) -> float:
        """Return current uninvested USDT balance."""
        return self._available_cash_usdt

    @property
    def active_position(self) -> Position | None:
        """Return currently open primary position or None."""
        self._reconcile_external_clear()
        return self._active_position

    @property
    def active_positions(self) -> tuple[Position, ...]:
        """Return all currently open positions across symbols."""
        self._reconcile_external_clear()
        return tuple(self._open_positions.values())

    def has_any_position(self) -> bool:
        """Return True when at least one open position exists."""
        self._reconcile_external_clear()
        return bool(self._open_positions)

    def has_position(self, symbol: str) -> bool:
        """Return True when an open position exists for the given symbol."""
        self._reconcile_external_clear()
        return symbol.upper() in self._open_positions

    def set_buy_blocked_symbols(self, symbols: set[str]) -> None:
        """Configure symbols prohibited from opening new BUY or DCA orders.

        Args:
            symbols: Set of uppercase symbol names blocked from new entries.
        """
        self._buy_blocked_symbols = {sym.upper() for sym in symbols}
        if self._swing_candidate_symbols:
            self._swing_candidate_symbols = {
                sym
                for sym in self._swing_candidate_symbols
                if sym not in self._buy_blocked_symbols
            }

    def set_swing_candidate_symbols(self, symbols: set[str]) -> None:
        """Configure symbols eligible to occupy the high-target SWING slot.

        Args:
            symbols: Set of uppercase symbol identifiers approved for Swing holds.
        """
        self._swing_candidate_symbols = {
            sym.upper()
            for sym in symbols
            if sym.upper() not in self._buy_blocked_symbols
        }

    def _count_positions_by_role(self) -> tuple[int, int]:
        """Return counts of currently open (SWING, SCALP) positions."""
        swing_count = sum(
            1
            for pos in self._open_positions.values()
            if pos.slot_role == SlotRole.SWING
        )
        scalp_count = len(self._open_positions) - swing_count
        return swing_count, scalp_count

    def _estimated_total_equity_usdt(self) -> float:
        """Estimate total portfolio value in USDT from cash plus open notionals."""
        open_notional = sum(pos.notional_usdt for pos in self._open_positions.values())
        return round(self._available_cash_usdt + open_notional, 6)

    def _build_open_positions_summary(self) -> list[tuple[str, float, str]]:
        """Build summary tuples (symbol, notional_usdt, role) for notifications."""
        return [
            (pos.symbol, pos.notional_usdt, pos.slot_role.value)
            for pos in self._open_positions.values()
        ]

    def _compute_position_exit_levels(self, pos: Position) -> tuple[float, float]:
        """Compute zero-loss safety floor and take-profit price for a position role."""
        if pos.slot_role == SlotRole.SWING:
            swing_target = (
                pos.target_net_profit_usdt
                if pos.target_net_profit_usdt > 0.0
                else self._risk_config.swing_target_net_profit_usdt
            )
            return compute_no_loss_exit_levels(
                entry_price=pos.entry_price,
                quantity=pos.quantity,
                notional_usdt=pos.notional_usdt,
                taker_fee_rate=self._exchange_config.taker_fee_rate,
                min_target_net_profit_usdt=swing_target,
                include_live_spread_cushion=self._live_client is not None,
                min_floor_net_profit_usdt=0.002,
            )
        scalp_target = (
            pos.target_net_profit_usdt
            if pos.target_net_profit_usdt > 0.0
            else self._risk_config.min_target_net_profit_usdt
        )
        return compute_no_loss_exit_levels(
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            notional_usdt=pos.notional_usdt,
            taker_fee_rate=self._exchange_config.taker_fee_rate,
            min_target_net_profit_usdt=scalp_target,
            include_live_spread_cushion=self._live_client is not None,
        )

    def _compute_breakeven_floor_price(self, pos: Position) -> float:
        """Compute pure fee-covered breakeven price strictly above entry_price."""
        floor_price, _ = compute_no_loss_exit_levels(
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            notional_usdt=pos.notional_usdt,
            taker_fee_rate=self._exchange_config.taker_fee_rate,
            min_target_net_profit_usdt=0.002,
            include_live_spread_cushion=self._live_client is not None,
            min_floor_net_profit_usdt=0.002,
        )
        return floor_price

    def resolve_next_slot_role(self, symbol: str) -> SlotRole | None:
        """Determine whether symbol can open a SWING or SCALP slot right now.

        Args:
            symbol: Candidate trading pair symbol.

        Returns:
            Assigned SlotRole if entry is permitted, otherwise None.
        """
        self._reconcile_external_clear()
        upper_sym = symbol.upper()
        if upper_sym in self._buy_blocked_symbols:
            return None
        if upper_sym in self._open_positions:
            return None

        max_slots = max(1, self._risk_config.max_concurrent_positions)
        if self._open_positions and not self._risk_config.allow_secondary_symbol_slot:
            return None
        if len(self._open_positions) >= max_slots:
            return None

        min_required_cash = self._exchange_config.min_notional_usdt + 0.50
        if (
            not self._risk_config.enable_dual_target_slots
            or self._risk_config.swing_slot_count <= 0
        ):
            if (
                not self._open_positions
                or self._available_cash_usdt >= min_required_cash
            ):
                return SlotRole.SCALP
            return None

        if self._available_cash_usdt < min_required_cash:
            return None

        swing_limit = min(self._risk_config.swing_slot_count, max_slots)
        scalp_limit = max(0, max_slots - swing_limit)
        open_swing, open_scalp = self._count_positions_by_role()

        excluded_swing = {
            sym.upper() for sym in self._risk_config.swing_excluded_symbols
        }
        is_swing_candidate = upper_sym not in excluded_swing and (
            not self._swing_candidate_symbols
            or upper_sym in self._swing_candidate_symbols
        )
        if open_swing < swing_limit and is_swing_candidate:
            return SlotRole.SWING

        if open_scalp < scalp_limit:
            remaining_swings = max(0, swing_limit - open_swing)
            if remaining_swings == 0:
                return SlotRole.SCALP
            total_equity = self._estimated_total_equity_usdt()
            reserved_swing = (
                remaining_swings
                * total_equity
                * self._risk_config.swing_allocation_fraction
            )
            scalp_pool = (self._available_cash_usdt * 0.96) - reserved_swing
            if scalp_pool >= min_required_cash:
                return SlotRole.SCALP
            if total_equity < (min_required_cash * 2.5) and open_scalp == 0:
                return SlotRole.SCALP

        return None

    def _compute_dual_slot_notional_usdt(self, slot_role: SlotRole) -> float:
        """Calculate asymmetric USDT budget for a SWING or SCALP slot."""
        min_notional = self._exchange_config.min_notional_usdt
        min_safe_slot = min_notional + 0.75
        max_slots = max(1, self._risk_config.max_concurrent_positions)
        swing_limit = min(self._risk_config.swing_slot_count, max_slots)
        scalp_limit = max(0, max_slots - swing_limit)
        open_swing, open_scalp = self._count_positions_by_role()
        total_equity = self._estimated_total_equity_usdt()
        target_swing = total_equity * self._risk_config.swing_allocation_fraction

        if slot_role == SlotRole.SWING:
            remaining_scalps = max(0, scalp_limit - open_scalp)
            reserved_for_scalps = remaining_scalps * (min_notional + 0.85)
            max_feasible = (self._available_cash_usdt * 0.96) - reserved_for_scalps
            if max_feasible >= min_safe_slot:
                proposed = min(target_swing, max_feasible)
                if remaining_scalps == 0:
                    proposed = min(
                        max(proposed, self._available_cash_usdt * 0.94),
                        self._available_cash_usdt * 0.96,
                    )
                return round(max(min_safe_slot, proposed), 4)
            return round(min(self._available_cash_usdt * 0.95, min_safe_slot), 4)

        remaining_scalps = max(1, scalp_limit - open_scalp)
        remaining_swings = max(0, swing_limit - open_swing)
        reserved_for_swings = remaining_swings * target_swing
        reserve_ratio = (
            0.95 if (remaining_scalps == 1 and remaining_swings == 0) else 0.96
        )
        available_for_scalps = max(
            0.0,
            (self._available_cash_usdt * reserve_ratio) - reserved_for_swings,
        )
        raw_scalp = available_for_scalps / remaining_scalps
        max_scalp_cap = max(min_safe_slot, total_equity * 0.34)
        capped_scalp = min(raw_scalp, max_scalp_cap)
        if capped_scalp < min_safe_slot and self._available_cash_usdt >= (
            min_notional + 0.50
        ):
            return round(min(self._available_cash_usdt * 0.95, min_safe_slot), 4)
        return round(capped_scalp, 4)

    def get_open_position(self, symbol: str) -> Position | None:
        """Return the open position for a specific symbol or None."""
        self._reconcile_external_clear()
        return self._open_positions.get(symbol.upper())

    def can_open_position_for_symbol(self, symbol: str) -> bool:
        """Return True if a new slot is available to open a position for symbol."""
        return self.resolve_next_slot_role(symbol) is not None

    def can_open_additional_positions(self) -> bool:
        """Return True if additional slots and cash remain for concurrent trades."""
        self._reconcile_external_clear()
        if not self._open_positions:
            return True
        if not self._risk_config.allow_secondary_symbol_slot:
            return False
        if len(self._open_positions) >= max(
            1, self._risk_config.max_concurrent_positions
        ):
            return False
        min_required_cash = self._exchange_config.min_notional_usdt + 0.50
        return self._available_cash_usdt >= min_required_cash

    def _can_dca_position(self, pos: Position, current_price: float) -> bool:
        """Return True if a held non-reconciled position qualifies for Smart DCA."""
        if pos.symbol.upper() in self._buy_blocked_symbols:
            return False
        if not self._risk_config.enable_dca_average_down:
            return False
        if pos.position_id.startswith("reconciled-"):
            return False
        if pos.position_id in self._dca_applied_position_ids:
            return False
        if pos.is_partially_closed:
            return False
        trigger_price = pos.entry_price * (
            1.0 - self._risk_config.dca_trigger_drawdown_ratio
        )
        if current_price > trigger_price:
            return False
        min_required_cash = self._exchange_config.min_notional_usdt + 0.50
        return self._available_cash_usdt >= min_required_cash

    def can_open_or_dca_symbol(self, symbol: str, current_price: float) -> bool:
        """Return True if symbol can open a fresh slot or execute a Smart DCA buy."""
        self._reconcile_external_clear()
        upper_sym = symbol.upper()
        if upper_sym in self._buy_blocked_symbols:
            return False
        pos = self._open_positions.get(upper_sym)
        if pos is None:
            return self.can_open_position_for_symbol(symbol)
        return self._can_dca_position(pos, current_price)

    def _resolve_target_position(self, symbol: str | None) -> Position | None:
        """Resolve either a symbol-scoped position or the primary active position."""
        self._reconcile_external_clear()
        if symbol is not None:
            return self._open_positions.get(symbol.upper())
        return self._active_position

    def _persist_position_update(self, position: Position) -> None:
        """Update cached position map, primary reference, and SQLite ledger."""
        self._open_positions[position.symbol.upper()] = position
        self._sync_primary_active_position()
        self._ledger.save_open_position_for_symbol(position)

    def evaluate_protective_exits(
        self,
        current_price: float,
        timestamp: datetime,
        symbol: str | None = None,
    ) -> TradeExecution | None:
        """Check whether an open position has triggered stop loss or take profit.

        Args:
            current_price: Latest market trade or close price.
            timestamp: UTC timestamp of the price observation.
            symbol: Optional trading symbol to scope exit checks cleanly.

        Returns:
            TradeExecution if the position was closed, otherwise None.
        """
        pos = self._resolve_target_position(symbol)
        if pos is None:
            return None

        new_peak = max(pos.peak_price, current_price, pos.entry_price)
        base_trough = pos.trough_price if pos.trough_price > 0.0 else pos.entry_price
        new_trough = min(base_trough, current_price)

        if self._risk_config.no_loss_spot_mode:
            safety_floor, role_take_profit = self._compute_position_exit_levels(pos)

            if pos.slot_role == SlotRole.SWING:
                effective_tp = max(pos.take_profit_price, role_take_profit)
                effective_floor = max(pos.stop_loss_price, safety_floor)
                swing_arm_price = round(
                    pos.entry_price
                    * (1.0 + self._risk_config.swing_trailing_arm_ratio),
                    10,
                )
                if (
                    self._risk_config.enable_swing_trailing_lock
                    and new_peak >= swing_arm_price
                ):
                    min_lock_floor, _ = compute_no_loss_exit_levels(
                        entry_price=pos.entry_price,
                        quantity=pos.quantity,
                        notional_usdt=pos.notional_usdt,
                        taker_fee_rate=self._exchange_config.taker_fee_rate,
                        min_target_net_profit_usdt=(
                            self._risk_config.swing_trailing_min_lock_profit_usdt
                        ),
                        include_live_spread_cushion=self._live_client is not None,
                    )
                    trail_floor = round(
                        new_peak
                        * (1.0 - self._risk_config.swing_trailing_pullback_ratio),
                        10,
                    )
                    effective_floor = max(
                        effective_floor,
                        safety_floor,
                        min_lock_floor,
                        trail_floor,
                    )

                if (
                    new_peak > pos.peak_price
                    or new_trough < base_trough
                    or pos.trough_price <= 0.0
                    or effective_floor > pos.stop_loss_price
                    or effective_tp > pos.take_profit_price
                ):
                    pos = replace(
                        pos,
                        stop_loss_price=effective_floor,
                        take_profit_price=effective_tp,
                        peak_price=new_peak,
                        trough_price=new_trough,
                    )
                    self._persist_position_update(pos)

                if current_price >= effective_tp and current_price >= safety_floor:
                    return self._close_active_position(
                        exit_price=current_price,
                        timestamp=timestamp,
                        reason=ExitReason.TAKE_PROFIT.value,
                        symbol=pos.symbol,
                    )

                if (
                    self._risk_config.enable_swing_trailing_lock
                    and new_peak >= swing_arm_price
                    and current_price <= effective_floor
                    and current_price >= safety_floor
                ):
                    return self._close_active_position(
                        exit_price=current_price,
                        timestamp=timestamp,
                        reason=ExitReason.TRAILING_STOP.value,
                        symbol=pos.symbol,
                    )

                return None

            effective_floor = max(pos.stop_loss_price, safety_floor)
            round_trip_fee = 2.0 * self._exchange_config.taker_fee_rate
            arm_price = max(
                round(safety_floor * 1.0008, 10),
                round(
                    pos.entry_price
                    * (
                        1.0
                        + round_trip_fee
                        + self._risk_config.no_loss_trailing_arm_ratio
                    ),
                    10,
                ),
            )
            if (
                self._risk_config.enable_no_loss_trailing_runner
                and new_peak >= arm_price
            ):
                pullback_ratio = self._risk_config.no_loss_trailing_pullback_ratio
                trail_floor = round(new_peak * (1.0 - pullback_ratio), 10)
                effective_floor = max(effective_floor, safety_floor, trail_floor)

            if (
                new_peak > pos.peak_price
                or new_trough < base_trough
                or pos.trough_price <= 0.0
                or effective_floor > pos.stop_loss_price
            ):
                pos = replace(
                    pos,
                    stop_loss_price=effective_floor,
                    peak_price=new_peak,
                    trough_price=new_trough,
                )
                self._persist_position_update(pos)

            if self._risk_config.enable_no_loss_trailing_runner:
                if (
                    new_peak >= arm_price
                    and current_price <= effective_floor
                    and current_price >= safety_floor
                ):
                    return self._close_active_position(
                        exit_price=current_price,
                        timestamp=timestamp,
                        reason=ExitReason.TRAILING_STOP.value,
                        symbol=pos.symbol,
                    )
                return None

            if current_price < effective_floor:
                return None

            if current_price >= pos.take_profit_price:
                return self._close_active_position(
                    exit_price=current_price,
                    timestamp=timestamp,
                    reason=ExitReason.TAKE_PROFIT.value,
                    symbol=pos.symbol,
                )

            if new_peak >= pos.take_profit_price and current_price >= effective_floor:
                return self._close_active_position(
                    exit_price=current_price,
                    timestamp=timestamp,
                    reason=ExitReason.TRAILING_STOP.value,
                    symbol=pos.symbol,
                )

            return None

        gain_ratio = (new_peak - pos.entry_price) / pos.entry_price
        current_gain = (current_price - pos.entry_price) / pos.entry_price

        if (
            self._risk_config.enable_partial_take_profit
            and not pos.is_partially_closed
            and current_gain >= self._risk_config.tp1_profit_ratio
        ):
            partial_exec = self._execute_partial_take_profit(
                current_price, timestamp, pos
            )
            if partial_exec is not None:
                return partial_exec

        if gain_ratio >= 0.0032:
            breakeven_floor = round(pos.entry_price * 1.0025, 10)
            trail_pct = 0.0018 if pos.is_partially_closed else 0.0008
            trailing_distance = round(pos.entry_price * trail_pct, 10)
            trailing_candidate = round(new_peak - trailing_distance, 10)
            new_sl = max(pos.stop_loss_price, breakeven_floor, trailing_candidate)
            if new_sl > pos.stop_loss_price or new_peak > pos.peak_price:
                pos = replace(pos, stop_loss_price=new_sl, peak_price=new_peak)
                self._persist_position_update(pos)
        elif gain_ratio >= 0.0022:
            breakeven_floor = round(pos.entry_price * 1.0022, 10)
            trail_pct = 0.0008
            trailing_distance = round(pos.entry_price * trail_pct, 10)
            trailing_candidate = round(new_peak - trailing_distance, 10)
            new_sl = max(pos.stop_loss_price, breakeven_floor, trailing_candidate)
            if new_sl > pos.stop_loss_price or new_peak > pos.peak_price:
                pos = replace(pos, stop_loss_price=new_sl, peak_price=new_peak)
                self._persist_position_update(pos)
        elif gain_ratio >= 0.0014:
            defensive_floor = round(pos.entry_price * 0.9988, 10)
            new_sl = max(pos.stop_loss_price, defensive_floor)
            if new_sl > pos.stop_loss_price or new_peak > pos.peak_price:
                pos = replace(pos, stop_loss_price=new_sl, peak_price=new_peak)
                self._persist_position_update(pos)
        elif new_peak > pos.peak_price:
            pos = replace(pos, peak_price=new_peak)
            self._persist_position_update(pos)

        updated_pos = self._open_positions.get(pos.symbol.upper(), pos)
        if current_price <= updated_pos.stop_loss_price:
            reason = (
                ExitReason.TRAILING_STOP.value
                if updated_pos.stop_loss_price >= updated_pos.entry_price
                else ExitReason.STOP_LOSS.value
            )
            return self._close_active_position(
                exit_price=current_price,
                timestamp=timestamp,
                reason=reason,
                symbol=updated_pos.symbol,
            )

        if current_price >= updated_pos.take_profit_price:
            return self._close_active_position(
                exit_price=current_price,
                timestamp=timestamp,
                reason=ExitReason.TAKE_PROFIT.value,
                symbol=updated_pos.symbol,
            )

        return None

    def _execute_partial_take_profit(
        self,
        current_price: float,
        timestamp: datetime,
        target_position: Position | None = None,
    ) -> TradeExecution | None:
        """Execute a partial take-profit order locking in profit on half the position.

        Args:
            current_price: Market fill price per unit of base asset.
            timestamp: UTC timestamp of partial exit execution.

        Returns:
            TradeExecution representing the partial SELL order, or None.
        """
        pos = (
            target_position
            if target_position is not None
            else self._active_position
        )
        if pos is None or pos.is_partially_closed:
            return None
        raw_qty_to_sell = pos.quantity * self._risk_config.tp1_close_fraction
        if self._live_client is not None:
            step_size = self._live_client.fetch_symbol_step_size(pos.symbol)
            formatted_qty = quantize_step_size(raw_qty_to_sell, step_size)
            sell_quantity = float(formatted_qty)
        else:
            sell_quantity = round(raw_qty_to_sell, 8)

        min_notional = self._exchange_config.min_notional_usdt
        if (sell_quantity * current_price) < min_notional:
            return None

        client_order_id = generate_client_order_id(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            timestamp=timestamp,
            reason=ExitReason.PARTIAL_TAKE_PROFIT.value,
        )
        if client_order_id in self._executed_order_ids:
            return None

        actual_exit_price = current_price
        gross_exit_notional_usdt = round(sell_quantity * current_price, 6)

        if self._live_client is not None:
            if self._live_client.is_cooldown_active():
                return None
            try:
                live_res = self._live_client.place_market_sell_by_quantity(
                    symbol=pos.symbol,
                    quantity=sell_quantity,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                logger.warning(
                    "Live partial SELL order deferred due to exchange error (%s); "
                    "will retry after cooldown.",
                    exc,
                )
                return None
            live_qty = float(live_res.get("executedQty", 0.0))
            live_quote = float(live_res.get("cummulativeQuoteQty", 0.0))
            if live_qty > 0.0 and live_quote > 0.0:
                actual_exit_price = live_quote / live_qty
                gross_exit_notional_usdt = round(live_quote, 6)
                sell_quantity = live_qty

        exit_fee_usdt = round(
            gross_exit_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        net_exit_proceeds_usdt = gross_exit_notional_usdt - exit_fee_usdt
        fraction_closed = sell_quantity / pos.quantity
        closed_notional = round(pos.notional_usdt * fraction_closed, 6)
        realized_pnl_usdt = round(net_exit_proceeds_usdt - closed_notional, 6)

        self._available_cash_usdt = round(
            self._available_cash_usdt + net_exit_proceeds_usdt, 6
        )
        if self._live_client is not None:
            self._live_client.update_known_usdt_balance(self._available_cash_usdt)
        self._realized_pnl_usdt = round(self._realized_pnl_usdt + realized_pnl_usdt, 6)
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + exit_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)

        remaining_quantity = round(pos.quantity - sell_quantity, 8)
        remaining_notional = round(pos.notional_usdt - closed_notional, 6)
        breakeven_floor = round(pos.entry_price * 1.0025, 8)
        tp2_price = round(
            pos.entry_price * (1.0 + self._risk_config.tp2_take_profit_ratio), 8
        )
        new_peak = max(pos.peak_price, current_price)
        trail_pct = 0.0035
        trailing_distance = round(pos.entry_price * trail_pct, 8)
        trailing_candidate = round(new_peak - trailing_distance, 8)
        new_sl = max(pos.stop_loss_price, breakeven_floor, trailing_candidate)

        updated_pos = replace(
            pos,
            quantity=remaining_quantity,
            notional_usdt=remaining_notional,
            stop_loss_price=new_sl,
            take_profit_price=tp2_price,
            is_partially_closed=True,
            peak_price=new_peak,
        )
        self._persist_position_update(updated_pos)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=pos.position_id,
            symbol=pos.symbol,
            side=OrderSide.SELL,
            executed_at=timestamp,
            execution_price=actual_exit_price,
            quantity=sell_quantity,
            gross_notional_usdt=gross_exit_notional_usdt,
            fee_usdt=exit_fee_usdt,
            realized_pnl_usdt=realized_pnl_usdt,
            reason=ExitReason.PARTIAL_TAKE_PROFIT.value,
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        logger.info(
            "Executed %s PARTIAL TAKE PROFIT (TP1) | symbol=%s price=%s qty=%s "
            "pnl=%.4f USDT remaining_qty=%s",
            mode_tag,
            execution.symbol,
            format_price(execution.execution_price),
            format_quantity(execution.quantity),
            realized_pnl_usdt,
            format_quantity(remaining_quantity),
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

    def process_signal(
        self,
        signal: MarketSignal,
        win_probability_pct: float | None = None,
        allocation_fraction: float | None = None,
    ) -> TradeExecution | None:
        """Process a strategy MarketSignal and execute paper orders if valid.

        Args:
            signal: Evaluated MarketSignal from the technical strategy layer.
            win_probability_pct: Optional 10,000-replay Time Machine win probability.
            allocation_fraction: Optional dynamic budget allocation fraction.

        Returns:
            TradeExecution if an entry or exit order was filled, otherwise None.
        """
        self._reconcile_external_clear()
        target_pos = self._open_positions.get(signal.symbol.upper())

        if target_pos is not None and signal.signal_type == SignalType.SELL:
            if self._risk_config.no_loss_spot_mode:
                safety_floor, role_take_profit = self._compute_position_exit_levels(
                    target_pos
                )
                if target_pos.slot_role == SlotRole.SWING:
                    effective_tp = max(target_pos.take_profit_price, role_take_profit)
                    if signal.reference_price >= effective_tp:
                        return self._close_active_position(
                            exit_price=signal.reference_price,
                            timestamp=signal.evaluated_at,
                            reason=ExitReason.SIGNAL_REVERSAL.value,
                            symbol=target_pos.symbol,
                        )
                    if self._risk_config.enable_drawdown_breakeven_escape:
                        trigger_low = target_pos.entry_price * (
                            1.0 - self._risk_config.drawdown_escape_trigger_ratio
                        )
                        if (
                            0.0 < target_pos.trough_price <= trigger_low
                            and signal.reference_price >= safety_floor
                        ):
                            return self._close_active_position(
                                exit_price=signal.reference_price,
                                timestamp=signal.evaluated_at,
                                reason=ExitReason.BREAKEVEN_RECOVERY.value,
                                symbol=target_pos.symbol,
                            )
                    return None
                min_profitable_exit = max(
                    target_pos.stop_loss_price,
                    safety_floor,
                )
                if self._risk_config.enable_no_loss_trailing_runner:
                    round_trip_fee = 2.0 * self._exchange_config.taker_fee_rate
                    arm_price = max(
                        round(safety_floor * 1.0008, 10),
                        round(
                            target_pos.entry_price
                            * (
                                1.0
                                + round_trip_fee
                                + self._risk_config.no_loss_trailing_arm_ratio
                            ),
                            10,
                        ),
                    )
                    if (
                        target_pos.peak_price >= arm_price
                        and signal.reference_price > target_pos.stop_loss_price
                    ):
                        return None
                if signal.reference_price >= min_profitable_exit:
                    return self._close_active_position(
                        exit_price=signal.reference_price,
                        timestamp=signal.evaluated_at,
                        reason=ExitReason.SIGNAL_REVERSAL.value,
                        symbol=target_pos.symbol,
                    )
                if self._risk_config.enable_drawdown_breakeven_escape:
                    trigger_low = target_pos.entry_price * (
                        1.0 - self._risk_config.drawdown_escape_trigger_ratio
                    )
                    breakeven_floor = self._compute_breakeven_floor_price(target_pos)
                    if (
                        0.0 < target_pos.trough_price <= trigger_low
                        and signal.reference_price >= breakeven_floor
                    ):
                        return self._close_active_position(
                            exit_price=signal.reference_price,
                            timestamp=signal.evaluated_at,
                            reason=ExitReason.BREAKEVEN_RECOVERY.value,
                            symbol=target_pos.symbol,
                        )
                return None
            min_ratio = max(
                self._risk_config.min_signal_exit_profit_ratio,
                (self._exchange_config.taker_fee_rate * 2.0)
                + self._risk_config.min_net_profit_ratio,
            )
            min_profitable_exit = target_pos.entry_price * (1.0 + min_ratio)
            if signal.reference_price >= min_profitable_exit:
                return self._close_active_position(
                    exit_price=signal.reference_price,
                    timestamp=signal.evaluated_at,
                    reason=ExitReason.SIGNAL_REVERSAL.value,
                    symbol=target_pos.symbol,
                )
            return None

        if signal.signal_type == SignalType.BUY:
            if target_pos is None and self.can_open_position_for_symbol(signal.symbol):
                return self._open_new_position(
                    signal=signal,
                    win_probability_pct=win_probability_pct,
                    allocation_fraction=allocation_fraction,
                )
            if target_pos is not None and self._can_dca_position(
                target_pos, signal.reference_price
            ):
                return self._average_down_position(
                    existing_pos=target_pos,
                    signal=signal,
                    win_probability_pct=win_probability_pct,
                )

        return None

    def _open_new_position(
        self,
        signal: MarketSignal,
        win_probability_pct: float | None = None,
        allocation_fraction: float | None = None,
    ) -> TradeExecution | None:
        """Validate and execute a simulated BUY entry order.

        Args:
            signal: BUY MarketSignal with entry, stop loss, and take profit prices.
            win_probability_pct: Optional 10,000-replay Time Machine win probability.
            allocation_fraction: Optional dynamic budget allocation fraction.

        Returns:
            TradeExecution if the order passed validation and filled, else None.
        """
        assigned_role = self.resolve_next_slot_role(signal.symbol)
        if assigned_role is None:
            return None

        use_dual_profile = (
            self._risk_config.enable_dual_target_slots
            and self._risk_config.swing_slot_count > 0
        )
        client_order_id = generate_client_order_id(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            timestamp=signal.evaluated_at,
            reason="SIGNAL_ENTRY",
            slot_role=assigned_role if use_dual_profile else None,
        )
        if client_order_id in self._executed_order_ids:
            return None

        if (
            self._live_client is not None
            and self._live_client.get_known_usdt_balance() is None
            and not self._live_client.is_cooldown_active()
            and hasattr(self._live_client, "fetch_available_usdt_balance")
        ):
            live_usdt = self._live_client.fetch_available_usdt_balance()
            if live_usdt is not None and live_usdt > 0.0:
                self._available_cash_usdt = round(live_usdt, 6)

        if use_dual_profile:
            proposed_notional_usdt = self._compute_dual_slot_notional_usdt(
                assigned_role
            )
        elif self._risk_config.use_dynamic_slot_sizing:
            remaining_slots = max(
                1,
                self._risk_config.max_concurrent_positions - len(self._open_positions),
            )
            reserve_ratio = 0.95 if remaining_slots == 1 else 0.96
            raw_slot_budget = (
                self._available_cash_usdt * reserve_ratio
            ) / remaining_slots
            min_safe_slot = self._exchange_config.min_notional_usdt + 0.75
            if (
                raw_slot_budget < min_safe_slot
                and self._available_cash_usdt
                >= (self._exchange_config.min_notional_usdt + 0.50)
            ):
                proposed_notional_usdt = round(
                    min(self._available_cash_usdt * 0.95, min_safe_slot),
                    4,
                )
            else:
                proposed_notional_usdt = round(raw_slot_budget, 4)
        else:
            if (
                self._risk_config.enable_tranche_split
                and self._available_cash_usdt
                >= self._risk_config.tranche_split_min_cash_usdt
            ):
                effective_alloc = self._risk_config.tranche_split_fraction
            elif allocation_fraction is not None and allocation_fraction > 0.0:
                effective_alloc = allocation_fraction
            else:
                effective_alloc = self._risk_config.position_allocation_fraction

            proposed_notional_usdt = round(
                self._available_cash_usdt * effective_alloc,
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

        entry_price = signal.reference_price
        stop_loss_price = signal.stop_loss_price
        take_profit_price = signal.take_profit_price
        entry_fee_usdt = round(
            proposed_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        net_notional_usdt = proposed_notional_usdt - entry_fee_usdt
        quantity = round(net_notional_usdt / entry_price, 8)
        cash_spent_usdt = proposed_notional_usdt

        if self._live_client is not None:
            if self._live_client.is_cooldown_active():
                return None
            try:
                live_res = self._live_client.place_market_buy_by_quote(
                    symbol=signal.symbol,
                    quote_amount_usdt=proposed_notional_usdt,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                logger.warning(
                    "Live BUY order deferred due to exchange error (%s); "
                    "will retry after cooldown.",
                    exc,
                )
                return None
            live_qty = float(live_res.get("executedQty", 0.0))
            live_quote = float(live_res.get("cummulativeQuoteQty", 0.0))
            if live_qty > 0.0 and live_quote > 0.0:
                fills = live_res.get("fills", [])
                base_asset = (
                    signal.symbol[:-4].upper()
                    if signal.symbol.upper().endswith("USDT")
                    else ""
                )
                base_commission = sum(
                    float(fill.get("commission", 0.0))
                    for fill in fills
                    if isinstance(fill, dict)
                    and str(fill.get("commissionAsset", "")).upper() == base_asset
                )
                bnb_commission = sum(
                    float(fill.get("commission", 0.0))
                    for fill in fills
                    if isinstance(fill, dict)
                    and str(fill.get("commissionAsset", "")).upper() == "BNB"
                )
                entry_price = live_quote / live_qty
                cash_spent_usdt = round(live_quote, 6)
                pre_base_bal = self._live_client.get_known_asset_balance(base_asset)
                total_base_after_buy = max(
                    0.0, pre_base_bal + live_qty - base_commission
                )
                if base_commission > 0.0 and total_base_after_buy < live_qty:
                    step_size = self._live_client.fetch_symbol_step_size(signal.symbol)
                    quantity = float(
                        quantize_step_size(total_base_after_buy, step_size)
                    )
                else:
                    quantity = live_qty
                proposed_notional_usdt = round(quantity * entry_price, 6)
                retained_base_delta = (live_qty - base_commission) - quantity
                if base_asset:
                    self._live_client.adjust_known_asset_balance(
                        base_asset, retained_base_delta
                    )
                if base_asset != "BNB" and bnb_commission > 0.0:
                    self._live_client.adjust_known_asset_balance("BNB", -bnb_commission)
                price_delta = entry_price - signal.reference_price
                stop_loss_price = round(signal.stop_loss_price + price_delta, 8)
                take_profit_price = round(signal.take_profit_price + price_delta, 8)

        target_net_profit_usdt = (
            self._risk_config.swing_target_net_profit_usdt
            if assigned_role == SlotRole.SWING
            else self._risk_config.min_target_net_profit_usdt
        )
        if self._risk_config.no_loss_spot_mode:
            stop_loss_price, take_profit_price = compute_no_loss_exit_levels(
                entry_price=entry_price,
                quantity=quantity,
                notional_usdt=proposed_notional_usdt,
                taker_fee_rate=self._exchange_config.taker_fee_rate,
                min_target_net_profit_usdt=target_net_profit_usdt,
                include_live_spread_cushion=self._live_client is not None,
                min_floor_net_profit_usdt=(
                    0.002 if assigned_role == SlotRole.SWING else None
                ),
            )

        position_id = f"pos-{client_order_id[-12:]}"

        self._available_cash_usdt = round(
            max(0.0, self._available_cash_usdt - cash_spent_usdt), 6
        )
        if self._live_client is not None:
            self._live_client.update_known_usdt_balance(self._available_cash_usdt)
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + entry_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)

        new_pos = Position(
            position_id=position_id,
            symbol=signal.symbol,
            opened_at=signal.evaluated_at,
            entry_price=entry_price,
            quantity=quantity,
            notional_usdt=proposed_notional_usdt,
            entry_fee_usdt=entry_fee_usdt,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            peak_price=entry_price,
            slot_role=assigned_role,
            target_net_profit_usdt=(
                target_net_profit_usdt if self._risk_config.no_loss_spot_mode else 0.0
            ),
            trough_price=entry_price,
        )
        self._persist_position_update(new_pos)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=position_id,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            executed_at=signal.evaluated_at,
            execution_price=entry_price,
            quantity=quantity,
            gross_notional_usdt=proposed_notional_usdt,
            fee_usdt=entry_fee_usdt,
            realized_pnl_usdt=0.0,
            reason="SIGNAL_ENTRY",
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        logger.info(
            "Executed %s BUY | symbol=%s role=%s price=%s qty=%s notional=%.4f USDT "
            "sl=%s tp=%s",
            mode_tag,
            execution.symbol,
            assigned_role.value,
            format_price(execution.execution_price),
            format_quantity(execution.quantity),
            execution.gross_notional_usdt,
            format_price(stop_loss_price),
            format_price(take_profit_price),
        )
        self._notifier.send_alert(
            format_execution_alert(
                execution=execution,
                available_cash_usdt=self._available_cash_usdt,
                cop_per_usdt_rate=self._capital_config.cop_per_usdt_rate,
                is_live_order=self._live_client is not None,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                win_probability_pct=win_probability_pct,
                slot_role=assigned_role if use_dual_profile else None,
                target_net_profit_usdt=(
                    target_net_profit_usdt if use_dual_profile else None
                ),
                open_positions_summary=(
                    self._build_open_positions_summary()
                    if len(self._open_positions) > 1 or use_dual_profile
                    else None
                ),
            )
        )
        return execution

    def _average_down_position(
        self,
        existing_pos: Position,
        signal: MarketSignal,
        win_probability_pct: float | None = None,
    ) -> TradeExecution | None:
        """Execute a Smart DCA buy tranche to lower an open position's average cost.

        Args:
            existing_pos: Currently held open position for signal.symbol.
            signal: Confirmed BUY signal on a rebound structure.
            win_probability_pct: Optional Time Machine win probability percentage.

        Returns:
            TradeExecution if the Smart DCA order succeeded, else None.
        """
        client_order_id = generate_client_order_id(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            timestamp=signal.evaluated_at,
            reason="SMART_DCA",
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
            logger.warning("Skipped Smart DCA BUY due to validation: %s", exc)
            return None

        add_price = signal.reference_price
        add_fee_usdt = round(
            proposed_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        add_qty = round((proposed_notional_usdt - add_fee_usdt) / add_price, 8)
        cash_spent_usdt = proposed_notional_usdt
        add_notional = proposed_notional_usdt

        if self._live_client is not None:
            if self._live_client.is_cooldown_active():
                return None
            try:
                live_res = self._live_client.place_market_buy_by_quote(
                    symbol=signal.symbol,
                    quote_amount_usdt=proposed_notional_usdt,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                logger.warning("Live Smart DCA BUY deferred (%s).", exc)
                return None
            live_qty = float(live_res.get("executedQty", 0.0))
            live_quote = float(live_res.get("cummulativeQuoteQty", 0.0))
            if live_qty > 0.0 and live_quote > 0.0:
                add_price = live_quote / live_qty
                cash_spent_usdt = round(live_quote, 6)
                add_qty = live_qty
                add_notional = round(add_qty * add_price, 6)

        combined_qty = round(existing_pos.quantity + add_qty, 8)
        combined_notional = round(existing_pos.notional_usdt + add_notional, 6)
        new_avg_entry = combined_notional / max(combined_qty, 1e-12)
        if self._risk_config.no_loss_spot_mode:
            new_sl, new_tp = compute_no_loss_exit_levels(
                entry_price=new_avg_entry,
                quantity=combined_qty,
                notional_usdt=combined_notional,
                taker_fee_rate=self._exchange_config.taker_fee_rate,
                min_target_net_profit_usdt=self._risk_config.min_target_net_profit_usdt,
                include_live_spread_cushion=self._live_client is not None,
            )
        else:
            new_sl = round(new_avg_entry * 0.995, 8)
            new_tp = round(new_avg_entry * 1.0075, 8)

        self._available_cash_usdt = round(
            max(0.0, self._available_cash_usdt - cash_spent_usdt), 6
        )
        if self._live_client is not None:
            self._live_client.update_known_usdt_balance(self._available_cash_usdt)
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + add_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)
        self._dca_applied_position_ids.add(existing_pos.position_id)

        updated_pos = replace(
            existing_pos,
            entry_price=new_avg_entry,
            quantity=combined_qty,
            notional_usdt=combined_notional,
            entry_fee_usdt=round(existing_pos.entry_fee_usdt + add_fee_usdt, 6),
            stop_loss_price=new_sl,
            take_profit_price=new_tp,
            peak_price=max(add_price, new_avg_entry),
        )
        self._persist_position_update(updated_pos)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=existing_pos.position_id,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            executed_at=signal.evaluated_at,
            execution_price=add_price,
            quantity=add_qty,
            gross_notional_usdt=add_notional,
            fee_usdt=add_fee_usdt,
            realized_pnl_usdt=0.0,
            reason="SMART_DCA",
        )
        self._ledger.record_execution(execution)
        self._notifier.send_alert(
            format_execution_alert(
                execution=execution,
                available_cash_usdt=self._available_cash_usdt,
                cop_per_usdt_rate=self._capital_config.cop_per_usdt_rate,
                is_live_order=self._live_client is not None,
                stop_loss_price=new_sl,
                take_profit_price=new_tp,
                win_probability_pct=win_probability_pct,
            )
        )
        return execution

    def execute_controlled_partial_unlock(
        self,
        symbol: str,
        fraction: float = 0.50,
        current_price: float | None = None,
        timestamp: datetime | None = None,
    ) -> TradeExecution | None:
        """Sell a controlled fraction of an open holding to unlock active cash.

        Args:
            symbol: Held trading pair symbol such as BTCUSDT.
            fraction: Fraction of held quantity to liquidate cleanly in Spot.
            current_price: Reference valuation price when running offline or paper.
            timestamp: UTC timestamp for ledger accounting.

        Returns:
            TradeExecution for the partial liquidation if performed, else None.
        """
        pos = self._open_positions.get(symbol.upper())
        if pos is None or not 0.0 < fraction < 1.0:
            return None

        client_order_id = f"unlock-{pos.symbol.lower()}-half-50pct"
        if client_order_id in self._executed_order_ids:
            return None

        if self._live_client is not None:
            step_size = self._live_client.fetch_symbol_step_size(pos.symbol)
            sell_qty = float(quantize_step_size(pos.quantity * fraction, step_size))
            remaining_qty = float(
                quantize_step_size(pos.quantity - sell_qty, step_size)
            )
        else:
            sell_qty = round(pos.quantity * fraction, 8)
            remaining_qty = round(pos.quantity - sell_qty, 8)

        if sell_qty <= 0.0 or (remaining_qty * pos.entry_price) < 4.5:
            return None

        exec_time = (
            timestamp if timestamp is not None else datetime.now(tz=timezone.utc)
        )
        actual_price = current_price if current_price is not None else pos.entry_price
        gross_proceeds = round(sell_qty * actual_price, 6)

        if self._live_client is not None:
            if self._live_client.is_cooldown_active():
                return None
            try:
                live_res = self._live_client.place_market_sell_by_quantity(
                    symbol=pos.symbol,
                    quantity=sell_qty,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                logger.warning(
                    "Controlled partial unlock on %s deferred (%s).",
                    pos.symbol,
                    exc,
                )
                return None
            live_qty = float(live_res.get("executedQty", 0.0))
            live_quote = float(live_res.get("cummulativeQuoteQty", 0.0))
            if live_qty > 0.0 and live_quote > 0.0:
                sell_qty = live_qty
                gross_proceeds = round(live_quote, 6)
                actual_price = live_quote / live_qty
                step_size = self._live_client.fetch_symbol_step_size(pos.symbol)
                remaining_qty = float(
                    quantize_step_size(max(0.0, pos.quantity - sell_qty), step_size)
                )

        exit_fee = round(gross_proceeds * self._exchange_config.taker_fee_rate, 6)
        net_proceeds = gross_proceeds - exit_fee
        sold_fraction = sell_qty / max(pos.quantity, 1e-12)
        closed_notional = round(pos.notional_usdt * sold_fraction, 6)
        realized_pnl = round(net_proceeds - closed_notional, 6)

        self._available_cash_usdt = round(
            self._available_cash_usdt + net_proceeds, 6
        )
        if self._live_client is not None:
            self._live_client._balance_cache.pop("USDT", None)
            try:
                refreshed_usdt = self._live_client.fetch_available_usdt_balance()
                if refreshed_usdt >= 0.0:
                    self._available_cash_usdt = round(refreshed_usdt, 6)
            except Exception:
                self._live_client.update_known_usdt_balance(self._available_cash_usdt)

        self._realized_pnl_usdt = round(self._realized_pnl_usdt + realized_pnl, 6)
        self._total_fees_paid_usdt = round(self._total_fees_paid_usdt + exit_fee, 6)
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)

        remaining_notional = round(remaining_qty * pos.entry_price, 6)
        if self._risk_config.no_loss_spot_mode:
            new_sl, new_tp = compute_no_loss_exit_levels(
                entry_price=pos.entry_price,
                quantity=remaining_qty,
                notional_usdt=remaining_notional,
                taker_fee_rate=self._exchange_config.taker_fee_rate,
                min_target_net_profit_usdt=self._risk_config.min_target_net_profit_usdt,
                include_live_spread_cushion=self._live_client is not None,
            )
        else:
            new_sl = pos.stop_loss_price
            new_tp = pos.take_profit_price

        updated_pos = replace(
            pos,
            quantity=remaining_qty,
            notional_usdt=remaining_notional,
            stop_loss_price=new_sl,
            take_profit_price=new_tp,
        )
        self._persist_position_update(updated_pos)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=pos.position_id,
            symbol=pos.symbol,
            side=OrderSide.SELL,
            executed_at=exec_time,
            execution_price=actual_price,
            quantity=sell_qty,
            gross_notional_usdt=gross_proceeds,
            fee_usdt=exit_fee,
            realized_pnl_usdt=realized_pnl,
            reason="PARTIAL_UNLOCK",
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        logger.info(
            "Executed %s PARTIAL UNLOCK | symbol=%s sold_qty=%s price=%s "
            "remaining_qty=%s freed_cash=%.4f USDT",
            mode_tag,
            pos.symbol,
            format_quantity(sell_qty),
            format_price(actual_price),
            format_quantity(remaining_qty),
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

    def _close_active_position(
        self,
        exit_price: float,
        timestamp: datetime,
        reason: str,
        symbol: str | None = None,
    ) -> TradeExecution | None:
        """Close the currently open position and realize net profit or loss.

        Args:
            exit_price: Market fill price per unit of base asset.
            timestamp: UTC timestamp of exit execution.
            reason: Exit trigger classifier string.
            symbol: Optional trading symbol to scope the exit.

        Returns:
            TradeExecution representing the closing SELL order.
        """
        position = self._resolve_target_position(symbol)
        if position is None:
            return None
        if self._risk_config.no_loss_spot_mode:
            if reason == ExitReason.BREAKEVEN_RECOVERY.value:
                safety_floor = self._compute_breakeven_floor_price(position)
            else:
                safety_floor, _ = self._compute_position_exit_levels(position)
            if exit_price < safety_floor:
                logger.info(
                    "Deferred exit on %s: price %s is below no-loss safety floor %s.",
                    position.symbol,
                    format_price(exit_price),
                    format_price(safety_floor),
                )
                return None
        client_order_id = generate_client_order_id(
            symbol=position.symbol,
            side=OrderSide.SELL,
            timestamp=timestamp,
            reason=reason,
        )
        if client_order_id in self._executed_order_ids:
            return None

        actual_exit_price = exit_price
        gross_exit_notional_usdt = round(position.quantity * exit_price, 6)
        sell_quantity = position.quantity

        if self._live_client is not None:
            if self._live_client.is_cooldown_active():
                return None
            try:
                live_res = self._live_client.place_market_sell_by_quantity(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    client_order_id=client_order_id,
                )
            except Exception as exc:
                logger.warning(
                    "Live SELL order deferred due to exchange error (%s); "
                    "will retry after cooldown.",
                    exc,
                )
                return None
            live_qty = float(live_res.get("executedQty", 0.0))
            live_quote = float(live_res.get("cummulativeQuoteQty", 0.0))
            if live_qty > 0.0 and live_quote > 0.0:
                actual_exit_price = live_quote / live_qty
                gross_exit_notional_usdt = round(live_quote, 6)
                sell_quantity = live_qty

        exit_fee_usdt = round(
            gross_exit_notional_usdt * self._exchange_config.taker_fee_rate, 6
        )
        net_exit_proceeds_usdt = gross_exit_notional_usdt - exit_fee_usdt
        realized_pnl_usdt = round(net_exit_proceeds_usdt - position.notional_usdt, 6)

        self._available_cash_usdt = round(
            self._available_cash_usdt + net_exit_proceeds_usdt, 6
        )
        if self._live_client is not None:
            self._live_client.update_known_usdt_balance(self._available_cash_usdt)
        self._realized_pnl_usdt = round(self._realized_pnl_usdt + realized_pnl_usdt, 6)
        self._total_fees_paid_usdt = round(
            self._total_fees_paid_usdt + exit_fee_usdt, 6
        )
        self._total_trades += 1
        self._executed_order_ids.add(client_order_id)
        self._open_positions.pop(position.symbol.upper(), None)
        self._sync_primary_active_position()
        self._ledger.clear_open_position_for_symbol(position.symbol)

        execution = TradeExecution(
            client_order_id=client_order_id,
            position_id=position.position_id,
            symbol=position.symbol,
            side=OrderSide.SELL,
            executed_at=timestamp,
            execution_price=actual_exit_price,
            quantity=sell_quantity,
            gross_notional_usdt=gross_exit_notional_usdt,
            fee_usdt=exit_fee_usdt,
            realized_pnl_usdt=realized_pnl_usdt,
            reason=reason,
        )
        self._ledger.record_execution(execution)
        mode_tag = "LIVE" if self._live_client is not None else "PAPER"
        use_dual_profile = (
            self._risk_config.enable_dual_target_slots
            and self._risk_config.swing_slot_count > 0
        )
        remaining_summary = self._build_open_positions_summary()
        logger.info(
            "Executed %s SELL | symbol=%s role=%s reason=%s price=%s "
            "net_pnl=%.4f USDT cash=%.4f USDT",
            mode_tag,
            execution.symbol,
            position.slot_role.value,
            execution.reason,
            format_price(execution.execution_price),
            execution.realized_pnl_usdt,
            self._available_cash_usdt,
        )
        self._notifier.send_alert(
            format_execution_alert(
                execution=execution,
                available_cash_usdt=self._available_cash_usdt,
                cop_per_usdt_rate=self._capital_config.cop_per_usdt_rate,
                is_live_order=self._live_client is not None,
                slot_role=position.slot_role if use_dual_profile else None,
                target_net_profit_usdt=(
                    position.target_net_profit_usdt if use_dual_profile else None
                ),
                open_positions_summary=remaining_summary or None,
            )
        )
        return execution

    def capture_portfolio_snapshot(
        self,
        current_price: float,
        timestamp: datetime,
        symbol: str | None = None,
        prices_by_symbol: dict[str, float] | None = None,
    ) -> PortfolioSnapshot:
        """Calculate and persist a mark-to-market portfolio valuation snapshot.

        Args:
            current_price: Latest market price used to value open holdings.
            timestamp: UTC timestamp of the valuation.
            symbol: Optional trading symbol associated with current_price.
            prices_by_symbol: Optional mapping of latest prices across symbols.

        Returns:
            Persisted PortfolioSnapshot domain record.
        """
        self._reconcile_external_clear()
        open_position_value_usdt = 0.0
        for pos_sym, pos in self._open_positions.items():
            mark_price = (
                prices_by_symbol[pos_sym]
                if prices_by_symbol is not None and pos_sym in prices_by_symbol
                else (
                    current_price
                    if (symbol is None or symbol.upper() == pos_sym)
                    and current_price > 0.0
                    else pos.entry_price
                )
            )
            gross_value = pos.quantity * mark_price
            estimated_exit_fee = gross_value * self._exchange_config.taker_fee_rate
            open_position_value_usdt = round(
                open_position_value_usdt + (gross_value - estimated_exit_fee), 6
            )

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
