"""Utility functions for cooldown logic."""

from typing import List


def is_soft_cooldown_triggered(pnls: List[float], lookback: int = 10, loss_count: int = 3):
    """Return True if number of losses in the last `lookback` trades >= `loss_count`."""
    if len(pnls) < lookback:
        return False, 0
    recent_losses = sum(1 for p in pnls[-lookback:] if p < 0)
    return recent_losses >= loss_count, recent_losses


def step_soft_cooldown(remaining_bars: int) -> int:
    """[Patch v5.0.20] Decrease soft cooldown bar counter.

    Args:
        remaining_bars (int): Current remaining cooldown bars.

    Returns:
        int: Updated remaining bars (never below zero).
    """
    if remaining_bars > 0:
        return remaining_bars - 1
    return 0
