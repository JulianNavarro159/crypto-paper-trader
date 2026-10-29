from __future__ import annotations


def format_price(price: float) -> str:
    """Format a numeric market price with adaptive precision for crypto assets.

    Args:
        price: Raw floating-point price value.

    Returns:
        String representation with decimal precision suited to asset magnitude.
    """
    abs_price = abs(price)
    if abs_price == 0.0:
        return "0.00"
    if abs_price < 0.0001:
        return f"{price:.8f}"
    if abs_price < 1.0:
        return f"{price:.6f}"
    if abs_price < 1000.0:
        return f"{price:.4f}"
    return f"{price:.2f}"


def format_quantity(quantity: float) -> str:
    """Format an asset trade quantity adapting to integer or fractional scales.

    Args:
        quantity: Raw floating-point asset quantity.

    Returns:
        Formatted quantity string without misleading trailing zeros.
    """
    abs_qty = abs(quantity)
    if abs_qty >= 1000.0:
        if quantity.is_integer():
            return f"{int(quantity):,}"
        return f"{quantity:,.2f}"
    if abs_qty >= 1.0:
        return f"{quantity:.4f}"
    return f"{quantity:.6f}"
