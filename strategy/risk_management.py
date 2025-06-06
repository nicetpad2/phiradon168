"""Simple risk management helpers."""
from __future__ import annotations


def calculate_position_size(balance: float, risk_pct: float, sl_distance: float, min_lot: float = 0.01) -> float:
    """Calculate lot size based on account balance and stop loss distance."""
    if balance <= 0 or risk_pct <= 0 or sl_distance <= 0:
        raise ValueError("balance, risk_pct and sl_distance must be positive")
    lot = balance * risk_pct / sl_distance
    return max(lot, min_lot)

__all__ = ["calculate_position_size"]
