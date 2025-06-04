"""Utility functions for cooldown logic."""

from typing import List


def is_soft_cooldown_triggered(pnls: List[float], lookback: int = 10, loss_count: int = 3):
    """[Patch v5.0.24] Determine if soft cooldown should activate.

    If fewer than ``lookback`` trades exist, all available PnLs are considered.
    This ensures early trades can still trigger the cooldown.
    """

    if not pnls:
        return False, 0

    effective_lookback = min(len(pnls), lookback)
    recent_losses = sum(1 for p in pnls[-effective_lookback:] if p < 0)
    return recent_losses >= loss_count, recent_losses


def step_soft_cooldown(remaining_bars: int, step: int = 1) -> int:
    """[Patch v5.0.24] Decrease soft cooldown bar counter.

    Args:
        remaining_bars (int): Current remaining cooldown bars.
        step (int): Number of bars to decrement per call.

    Returns:
        int: Updated remaining bars (never below zero).
    """
    if remaining_bars > 0:
        return max(0, remaining_bars - step)
    return 0
