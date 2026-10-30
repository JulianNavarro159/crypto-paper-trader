from __future__ import annotations

import logging
import os
import time
from typing import Protocol

import httpx

from crypto_paper_trader.domain import OrderSide, TradeExecution
from crypto_paper_trader.formatting import format_price, format_quantity

logger = logging.getLogger(__name__)


class NotificationService(Protocol):
    """Protocol for dispatching real-time trading alerts to mobile devices."""

    def send_alert(self, message: str) -> bool:
        """Send a formatted text alert to the configured recipient."""


class NullNotificationService:
    """No-op notification service used when mobile alerts are disabled."""

    def send_alert(self, message: str) -> bool:
        """Ignore alert message and return False.

        Args:
            message: Alert text payload.

        Returns:
            Always False indicating no external message was sent.
        """
        return False


class NtfyNotificationService:
    """Free registration-less mobile push notifier powered by ntfy.sh.

    Args:
        topic: Unique secret topic name subscribed in the ntfy mobile app.
        base_url: Base HTTP URL for the ntfy server.
        timeout_seconds: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        topic: str,
        base_url: str = "https://ntfy.sh",
        timeout_seconds: float = 8.0,
    ) -> None:
        self._topic = topic.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def send_alert(self, message: str) -> bool:
        """Push a real-time alert to the ntfy mobile application with 429 retries.

        Args:
            message: Plain-text message content to deliver.

        Returns:
            True if the ntfy server accepted the push, False otherwise.
        """
        if not self._topic:
            return False

        endpoint = f"{self._base_url}/{self._topic}"
        headers = {
            "Title": "Binance Spot Trading Bot",
            "Priority": "high",
            "Tags": "chart_with_upwards_trend",
        }
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        endpoint,
                        content=message.encode("utf-8"),
                        headers=headers,
                    )
                    if response.status_code == 429 and attempt < 3:
                        retry_after_str = response.headers.get("Retry-After", "")
                        wait_sec = (
                            float(retry_after_str)
                            if retry_after_str.isdigit()
                            else float(attempt * 3.0)
                        )
                        logger.warning(
                            "ntfy 429 rate limit; retrying in %.1fs (try %d/3)",
                            wait_sec,
                            attempt,
                        )
                        time.sleep(wait_sec)
                        continue
                    response.raise_for_status()
                return True
            except httpx.HTTPError as exc:
                if attempt < 3:
                    time.sleep(attempt * 2.0)
                    continue
                logger.warning("Failed to deliver ntfy mobile alert: %s", exc)
                return False
        return False


class DiscordNotificationService:
    """Discord Webhook client for free mobile and desktop trade notifications.

    Args:
        webhook_url: Full Discord channel webhook URL.
        timeout_seconds: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._webhook_url = webhook_url.strip()
        self._timeout_seconds = timeout_seconds

    def send_alert(self, message: str) -> bool:
        """Send a trading alert message to a Discord webhook channel.

        Args:
            message: Plain-text message content to deliver.

        Returns:
            True if Discord accepted the webhook payload, False otherwise.
        """
        if not self._webhook_url:
            return False

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(
                    self._webhook_url,
                    json={"content": f"```text\n{message}\n```"},
                )
                response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Failed to deliver Discord webhook alert: %s", exc)
            return False


class MultiChannelNotificationService:
    """Composite notifier that broadcasts alerts across all configured channels.

    Args:
        channels: Sequence of active NotificationService implementations.
    """

    def __init__(self, channels: list[NotificationService]) -> None:
        self._channels = channels

    def send_alert(self, message: str) -> bool:
        """Broadcast an alert to every configured mobile notification channel.

        Args:
            message: Plain-text message content to deliver.

        Returns:
            True if at least one channel delivered the message successfully.
        """
        delivered = False
        for channel in self._channels:
            if channel.send_alert(message):
                delivered = True
        return delivered


def build_notifier_from_env() -> NotificationService:
    """Construct active mobile notification services from environment variables.

    Returns:
        MultiChannelNotificationService if any channel is configured, else Null.
    """
    channels: list[NotificationService] = []

    ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
    if ntfy_topic and not ntfy_topic.startswith("your_"):
        logger.info("Mobile ntfy push notifications enabled (topic=%s)", ntfy_topic)
        channels.append(NtfyNotificationService(topic=ntfy_topic))

    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if discord_webhook and not discord_webhook.startswith("your_"):
        logger.info("Mobile Discord webhook notifications enabled")
        channels.append(DiscordNotificationService(webhook_url=discord_webhook))

    if not channels:
        return NullNotificationService()
    return MultiChannelNotificationService(channels=channels)


def translate_exit_reason(reason: str) -> str:
    """Translate an internal exit reason code into an easy-to-understand Spanish label.

    Args:
        reason: Internal exit trigger identifier string.

    Returns:
        Clear, non-technical Spanish explanation of why the sale was triggered.
    """
    translations = {
        "TAKE_PROFIT": "Meta de ganancia cumplida con exito",
        "PARTIAL_TAKE_PROFIT": "Primera meta alcanzada (50% asegurado en efectivo)",
        "STOP_LOSS": "Proteccion activada para cuidar tu dinero",
        "SIGNAL_REVERSAL": "Mercado sin fuerza; venta para asegurar ganancias",
        "TRAILING_STOP": "Ganancia protegida antes de un cambio de tendencia",
    }
    return translations.get(reason, "Venta completada de forma segura")


def format_execution_alert(
    execution: TradeExecution,
    available_cash_usdt: float,
    cop_per_usdt_rate: float,
    is_live_order: bool = False,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> str:
    """Format a TradeExecution into a friendly Spanish mobile notification.

    Args:
        execution: Completed buy or sell TradeExecution record.
        available_cash_usdt: Remaining uninvested USDT balance.
        cop_per_usdt_rate: Exchange rate from USDT to Colombian Pesos.
        is_live_order: True if executed on real Binance Spot account, False for paper.
        stop_loss_price: Optional protective stop loss price for BUY alerts.
        take_profit_price: Optional target take profit price for BUY alerts.

    Returns:
        Multi-line plain Spanish summary of trade details and portfolio balances.
    """
    mode_label = "BINANCE REAL" if is_live_order else "SIMULACION"
    base_asset = (
        execution.symbol[:-4]
        if execution.symbol.upper().endswith("USDT")
        else execution.symbol
    )
    cash_cop = available_cash_usdt * cop_per_usdt_rate

    if execution.side == OrderSide.BUY:
        coin_usdt = execution.gross_notional_usdt - execution.fee_usdt
        coin_cop = coin_usdt * cop_per_usdt_rate
        total_usdt = available_cash_usdt + coin_usdt
        total_cop = total_usdt * cop_per_usdt_rate
        targets_block = ""
        if take_profit_price is not None and stop_loss_price is not None:
            tp_str = format_price(take_profit_price)
            sl_str = format_price(stop_loss_price)
            targets_block = (
                f"\n--- PLAN DE ESTA OPERACION ---\n"
                f"Meta de ganancia: Vender al llegar a {tp_str} USDT\n"
                f"Piso de seguridad: Vender a {sl_str} USDT para cuidar tu dinero"
            )
        return (
            f"[{mode_label}] COMPRA DE {base_asset} REALIZADA CON EXITO\n"
            f"El bot detecto una buena oportunidad y compro {base_asset}.\n"
            f"Precio de compra: {format_price(execution.execution_price)} USDT\n"
            f"Monedas adquiridas: {format_quantity(execution.quantity)} {base_asset}\n"
            f"Inversion en esta compra: {coin_usdt:.2f} USDT (~{coin_cop:,.0f} COP)\n"
            f"--- ESTADO DE TU DINERO ---\n"
            f"Invertido en {base_asset}: {coin_usdt:.2f} USDT (~{coin_cop:,.0f} COP)\n"
            f"Efectivo libre en cuenta: {available_cash_usdt:.2f} USDT "
            f"(~{cash_cop:,.0f} COP)\n"
            f"BALANCE TOTAL: {total_usdt:.2f} USDT (~{total_cop:,.0f} COP)"
            f"{targets_block}"
        )

    pnl_cop = execution.realized_pnl_usdt * cop_per_usdt_rate
    sign = "+" if execution.realized_pnl_usdt >= 0 else ""

    if "PARTIAL" in execution.reason:
        return (
            f"[{mode_label}] PRIMERA META ALCANZADA: GANANCIA ASEGURADA\n"
            f"Moneda: {base_asset}\n"
            f"¿Que hizo el bot?: Vendio la mitad (50%) de tus {base_asset} "
            f"para guardar la ganancia en efectivo en tu cuenta.\n"
            f"Precio de venta: {format_price(execution.execution_price)} USDT\n"
            f"Monedas vendidas: {format_quantity(execution.quantity)} {base_asset}\n"
            f"Ganancia asegurada en efectivo: {sign}{execution.realized_pnl_usdt:.4f} "
            f"USDT ({sign}{pnl_cop:,.0f} COP)\n"
            f"--- ESTADO DE TU DINERO ---\n"
            f"Efectivo disponible en cuenta: {available_cash_usdt:.2f} USDT "
            f"(~{cash_cop:,.0f} COP)\n"
            f"Seguridad: El piso de proteccion se subio al precio de compra. "
            f"El 50% restante sigue activo sin ningun riesgo de perdida."
        )

    reason_label = translate_exit_reason(execution.reason)
    return (
        f"[{mode_label}] VENTA DE {base_asset} REALIZADA\n"
        f"Moneda: {base_asset}\n"
        f"Motivo: {reason_label}\n"
        f"Precio de venta: {format_price(execution.execution_price)} USDT\n"
        f"Monedas vendidas: {format_quantity(execution.quantity)} {base_asset}\n"
        f"Resultado de esta operacion: {sign}{execution.realized_pnl_usdt:.4f} USDT "
        f"({sign}{pnl_cop:,.0f} COP)\n"
        f"--- ESTADO DE TU DINERO ---\n"
        f"Dinero en efectivo: {available_cash_usdt:.2f} USDT (~{cash_cop:,.0f} COP)\n"
        f"BALANCE TOTAL: {available_cash_usdt:.2f} USDT (~{cash_cop:,.0f} COP)\n"
        f"Monedas abiertas: Ninguna (100% de tu dinero esta seguro en efectivo "
        f"esperando la proxima oportunidad)."
    )


def format_recovery_alert(error: Exception, sleep_seconds: float) -> str:
    """Format an unexpected runtime issue into a reassuring, non-technical alert.

    Args:
        error: Caught runtime exception instance.
        sleep_seconds: Cooldown sleep duration before resuming.

    Returns:
        Reassuring Spanish notification explaining safety and auto-recovery.
    """
    error_str = str(error)
    error_type = type(error).__name__
    minutes = max(1, int(sleep_seconds // 60))

    if (
        "418" in error_str
        or "MarketDataIngestionError" in error_type
        or "Rate limited" in error_str
        or "429" in error_str
    ):
        return (
            "PAUSA PREVENTIVA POR ALTO TRAFICO EN BINANCE\n"
            "Tu dinero esta 100% seguro en tu cuenta.\n"
            "Los servidores publicos de Binance tienen alta congestion temporal. "
            f"El bot hara una breve pausa de {minutes} minuto(s) para no sobrecargar "
            "la conexion y continuara vigilando el mercado de forma automatica."
        )

    is_network_issue = (
        isinstance(error, (ConnectionError, TimeoutError, OSError))
        or "Connect" in error_type
        or "Timeout" in error_type
        or "Network" in error_str
    )
    if is_network_issue:
        return (
            "PAUSA PREVENTIVA POR INTERMITENCIA DE RED\n"
            "Tu dinero esta 100% seguro en tu cuenta.\n"
            "Hubo una breve intermitencia en la conexion de internet. "
            f"El bot se reconectara de forma automatica en {int(sleep_seconds)} "
            "segundos."
        )

    return (
        "SISTEMA DE PROTECCION Y AUTO-RECUPERACION\n"
        "Tu dinero esta 100% seguro en tu cuenta.\n"
        "El bot detecto una variacion temporal en la plataforma y se esta "
        f"reiniciando de manera segura. Continuara trabajando en "
        f"{int(sleep_seconds)} segundos."
    )


def format_heartbeat_alert(
    symbol: str,
    interval: str,
    open_position_symbol: str | None,
    open_position_entry_price: float | None,
    available_cash_usdt: float | None,
    cop_per_usdt_rate: float,
    total_realized_pnl_usdt: float,
    closed_trades_count: int,
    win_rate_pct: float,
) -> str:
    """Format a two-hour health status report into clear, human-friendly Spanish.

    Args:
        symbol: Monitored market symbol.
        interval: Active candle timeframe.
        open_position_symbol: Symbol of active open holding or None.
        open_position_entry_price: Entry price of active holding or None.
        available_cash_usdt: Current uninvested cash in USDT or None.
        cop_per_usdt_rate: Exchange rate to Colombian Pesos.
        total_realized_pnl_usdt: Cumulative realized profit or loss.
        closed_trades_count: Number of closed trades executed.
        win_rate_pct: Percentage of winning trades.

    Returns:
        Clear status notification understandable without trading knowledge.
    """
    clean_symbol = symbol[:-4] if symbol.upper().endswith("USDT") else symbol

    if open_position_symbol and open_position_entry_price is not None:
        pos_clean = (
            open_position_symbol[:-4]
            if open_position_symbol.upper().endswith("USDT")
            else open_position_symbol
        )
        pos_line = (
            f"Operacion actual: Comprado en {pos_clean} a "
            f"{format_price(open_position_entry_price)} USDT. "
            f"El bot vigila el precio para vender con ganancia."
        )
    else:
        pos_line = (
            "Operacion actual: Ninguna. Tu dinero esta 100% seguro en "
            "efectivo esperando una buena oportunidad de compra."
        )

    cash_line = ""
    if available_cash_usdt is not None:
        cash_cop = available_cash_usdt * cop_per_usdt_rate
        cash_line = (
            f"\nDinero en efectivo: {available_cash_usdt:.2f} USDT "
            f"(~{cash_cop:,.0f} COP)"
        )

    perf_line = ""
    if closed_trades_count > 0:
        pnl_cop = total_realized_pnl_usdt * cop_per_usdt_rate
        sign = "+" if total_realized_pnl_usdt >= 0 else ""
        perf_line = (
            f"\nGanancia acumulada: {sign}{total_realized_pnl_usdt:.4f} USDT "
            f"({sign}{pnl_cop:,.0f} COP) en {closed_trades_count} "
            f"operaciones ({win_rate_pct:.0f}% exitosas)"
        )

    return (
        "REPORTE PERIODICO DE ESTADO 24/7\n"
        "Tu bot esta funcionando correctamente en la nube 24/7.\n"
        f"Moneda vigilada: {clean_symbol} (grafico de {interval})\n"
        f"{pos_line}"
        f"{cash_line}"
        f"{perf_line}"
    )


def format_startup_alert(
    mode: str,
    symbol: str,
    interval: str,
) -> str:
    """Format bot startup notification into intuitive Spanish.

    Args:
        mode: Bot operational mode.
        symbol: Target trading pair symbol.
        interval: Active candle timeframe.

    Returns:
        Spanish alert confirming the bot is active and analyzing the market.
    """
    clean_symbol = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
    mode_map = {
        "live": "Operaciones reales con dinero real en Binance",
        "paper": "Simulacion en tiempo real (dinero virtual)",
        "backtest": "Prueba historica de estrategia",
        "scan": "Escaneo de oportunidades de mercado",
    }
    mode_label = mode_map.get(mode.lower(), mode.upper())
    return (
        "BOT DE TRADING INICIADO CON EXITO\n"
        f"Modo: {mode_label}\n"
        f"Moneda vigilada: {clean_symbol} (grafico de {interval})\n"
        "El bot ya esta analizando el mercado continuamente para encontrar "
        "las mejores oportunidades de ganancia."
    )


def format_rotation_alert(
    old_symbol: str,
    new_symbol: str,
    interval: str,
) -> str:
    """Format market rotation notification into clear Spanish.

    Args:
        old_symbol: Previously monitored coin pair.
        new_symbol: Newly selected coin pair with higher potential.
        interval: Active candle timeframe.

    Returns:
        Clear explanation of why the monitored currency changed.
    """
    clean_old = (
        old_symbol[:-4] if old_symbol.upper().endswith("USDT") else old_symbol
    )
    clean_new = (
        new_symbol[:-4] if new_symbol.upper().endswith("USDT") else new_symbol
    )
    return (
        "CAMBIO AUTOMATICO DE MONEDA VIGILADA\n"
        f"El bot analizo el mercado y encontro que {clean_new} tiene una mejor "
        f"oportunidad de ganancia que {clean_old}.\n"
        f"A partir de este momento estara vigilando {clean_new} ({interval}) "
        "para buscar la mejor entrada."
    )


def format_balance_summary_alert(
    available_usdt: float,
    cop_per_usdt_rate: float,
    open_position_coin: str | None = None,
    open_position_qty: float = 0.0,
    open_position_usdt: float = 0.0,
    open_entry_price: float = 0.0,
    open_take_profit: float = 0.0,
) -> str:
    """Format an account balance snapshot into an intuitive Spanish summary.

    Args:
        available_usdt: Uninvested cash in USDT.
        cop_per_usdt_rate: Exchange rate to Colombian Pesos.
        open_position_coin: Optional coin name of open holding.
        open_position_qty: Quantity of held coin.
        open_position_usdt: Valuation in USDT of held coin.
        open_entry_price: Purchase price of held coin.
        open_take_profit: Profit target price of held coin.

    Returns:
        User-friendly Spanish summary of total account balance.
    """
    cash_cop = available_usdt * cop_per_usdt_rate
    total_usdt = available_usdt + open_position_usdt
    total_cop = total_usdt * cop_per_usdt_rate

    position_line = ""
    if open_position_coin and open_position_usdt > 0.0:
        coin_cop = open_position_usdt * cop_per_usdt_rate
        position_line = (
            f"Invertido en {open_position_coin}: {open_position_usdt:.2f} USDT "
            f"(~{coin_cop:,.0f} COP)\n"
            f"Monedas: {format_quantity(open_position_qty)} {open_position_coin}\n"
            f"Precio de compra: {format_price(open_entry_price)} USDT | "
            f"Meta de ganancia: {format_price(open_take_profit)} USDT\n"
        )

    return (
        "RESUMEN DE TU CUENTA EN BINANCE\n"
        f"Efectivo libre (USDT): {available_usdt:.2f} USDT (~{cash_cop:,.0f} COP)\n"
        f"{position_line}"
        f"BALANCE TOTAL: {total_usdt:.2f} USDT (~{total_cop:,.0f} COP)"
    )
