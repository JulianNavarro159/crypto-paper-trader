from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx

from crypto_paper_trader.domain import OrderSide, TradeExecution

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
        """Push a real-time alert to the ntfy mobile application.

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
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(
                    endpoint,
                    content=message.encode("utf-8"),
                    headers=headers,
                )
                response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Failed to deliver ntfy mobile alert: %s", exc)
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
    """Translate an internal exit reason code into a user-friendly Spanish label.

    Args:
        reason: Internal exit trigger identifier string.

    Returns:
        Descriptive Spanish explanation of the exit trigger.
    """
    translations = {
        "TAKE_PROFIT": "META DE GANANCIA ALCANZADA (TAKE PROFIT)",
        "STOP_LOSS": "LIMITE DE PROTECCION ACTIVADO (STOP LOSS)",
        "SIGNAL_REVERSAL": "TOMA DE GANANCIA POR SOBRECOMPRA",
        "TRAILING_STOP": "GANANCIA PROTEGIDA (TRAILING STOP)",
    }
    return translations.get(reason, reason)


def format_execution_alert(
    execution: TradeExecution,
    available_cash_usdt: float,
    cop_per_usdt_rate: float,
    is_live_order: bool = False,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
) -> str:
    """Format a TradeExecution into a Spanish mobile notification summary.

    Args:
        execution: Completed buy or sell TradeExecution record.
        available_cash_usdt: Remaining uninvested USDT balance.
        cop_per_usdt_rate: Exchange rate from USDT to Colombian Pesos.
        is_live_order: True if executed on real Binance Spot account, False for paper.
        stop_loss_price: Optional protective stop loss price for BUY alerts.
        take_profit_price: Optional target take profit price for BUY alerts.

    Returns:
        Multi-line Spanish summary of trade details and portfolio balances.
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
            targets_block = (
                f"\nMeta de venta (Ganancia): {take_profit_price:.4f} USDT"
                f"\nPrecio de proteccion (Stop Loss): {stop_loss_price:.4f} USDT"
            )
        return (
            f"[{mode_label}] COMPRA DE {base_asset} REALIZADA\n"
            f"Precio de compra: {execution.execution_price:.4f} USDT\n"
            f"Moneda comprada: {execution.quantity:.4f} {base_asset}\n"
            f"--- TU BALANCE ACTUAL ---\n"
            f"En moneda ({base_asset}): {coin_usdt:.2f} USDT ({coin_cop:,.0f} COP)\n"
            f"En efectivo (USDT): {available_cash_usdt:.2f} USDT "
            f"({cash_cop:,.0f} COP)\n"
            f"BALANCE TOTAL: {total_usdt:.2f} USDT ({total_cop:,.0f} COP)"
            f"{targets_block}"
        )

    pnl_cop = execution.realized_pnl_usdt * cop_per_usdt_rate
    reason_label = translate_exit_reason(execution.reason)
    return (
        f"[{mode_label}] VENTA DE {base_asset} REALIZADA\n"
        f"Motivo: {reason_label}\n"
        f"Precio de venta: {execution.execution_price:.4f} USDT\n"
        f"Resultado de esta operacion: {execution.realized_pnl_usdt:+.4f} USDT "
        f"({pnl_cop:+,.0f} COP)\n"
        f"--- TU BALANCE ACTUAL ---\n"
        f"En moneda ({base_asset}): 0.0000 {base_asset} (0 COP)\n"
        f"En efectivo (USDT): {available_cash_usdt:.2f} USDT "
        f"({cash_cop:,.0f} COP)\n"
        f"BALANCE TOTAL: {available_cash_usdt:.2f} USDT ({cash_cop:,.0f} COP)"
    )
