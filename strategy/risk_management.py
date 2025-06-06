"""Risk management helpers for position sizing."""
from typing import Tuple

__all__ = ["calculate_position_size"]


def calculate_position_size(balance: float, risk_pct: float, sl_distance: float) -> float:
    """Compute lot size based on risk percent and stop loss distance."""
    if sl_distance <= 0 or balance <= 0:
        return 0.0
    risk_amount = balance * risk_pct
    size = risk_amount / sl_distance
    return max(size, 0.0)
