"""Utilities for executing a single trade."""
from __future__ import annotations


def execute_order(order: dict, exit_price: float) -> float:
    """Close an order and return profit in price units."""
    if "sl" not in order or "tp" not in order:
        raise KeyError("order must contain sl and tp")
    if exit_price is None:
        raise ValueError("exit_price required")
    side = order.get("side")
    entry = order.get("entry_price")
    multiplier = 1 if side == "BUY" else -1
    return (exit_price - entry) * multiplier

__all__ = ["execute_order"]
