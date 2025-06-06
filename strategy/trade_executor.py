"""Execute trades using order and risk modules."""
from typing import Dict

from .order_management import place_order

__all__ = ["open_trade"]


def open_trade(side: str, price: float, sl: float, tp: float, size: float) -> Dict:
    """Open a trade and return the order dictionary."""
    return place_order(side, price, sl, tp, size)
