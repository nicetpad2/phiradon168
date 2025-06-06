"""Simplified order creation utilities."""
from __future__ import annotations


def create_order(side: str, price: float, sl: float, tp: float) -> dict:
    """Return a dictionary representing an order."""
    if sl is None or tp is None:
        raise ValueError("SL and TP must be provided")
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    return {
        "side": side,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
    }

__all__ = ["create_order"]
