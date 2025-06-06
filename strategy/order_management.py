"""Simple order placement utilities."""
from typing import Dict

__all__ = ["place_order"]


def place_order(side: str, price: float, sl: float, tp: float, size: float) -> Dict:
    """Return an order dictionary with mandatory SL/TP."""
    return {
        "side": side,
        "entry_price": price,
        "sl_price": sl,
        "tp_price": tp,
        "size": size,
    }
