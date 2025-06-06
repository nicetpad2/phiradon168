"""Wrapper utilities for stop-loss and take-profit calculations."""
from typing import Tuple

from src.money_management import atr_sl_tp

__all__ = ["atr_sl_tp_wrapper"]


def atr_sl_tp_wrapper(entry_price: float, atr: float, side: str, rr_ratio: float = 2.0) -> Tuple[float, float]:
    """Proxy to src.money_management.atr_sl_tp."""
    return atr_sl_tp(entry_price, atr, side, rr_ratio)
