"""Utility functions for cooldown logic."""

from typing import List
from dataclasses import dataclass


@dataclass
class CooldownState:
    """[Patch] Track cooldown and warning state within a fold."""

    consecutive_losses: int = 0
    drawdown_pct: float = 0.0
    warned_losses: bool = False
    warned_drawdown: bool = False
    cooldown_bars_remaining: int = 0


def is_soft_cooldown_triggered(pnls: List[float], lookback: int = 15, loss_count: int = 2):
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


def update_losses(state: CooldownState, pnl: float) -> int:
    """[Patch] Update consecutive loss count based on PnL."""
    if pnl < 0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0
    return state.consecutive_losses


def update_drawdown(state: CooldownState, drawdown_pct: float) -> float:
    """[Patch] Update current drawdown percentage."""
    state.drawdown_pct = drawdown_pct
    return state.drawdown_pct


def should_enter_cooldown(state: CooldownState, loss_thresh: int, dd_thresh: float) -> bool:
    """[Patch] Determine if soft cooldown should start."""
    return state.consecutive_losses >= loss_thresh or state.drawdown_pct >= dd_thresh


def enter_cooldown(state: CooldownState, lookback: int) -> int:
    """[Patch] Begin a soft cooldown period."""
    state.cooldown_bars_remaining = lookback
    return state.cooldown_bars_remaining


def should_warn_drawdown(state: CooldownState, threshold: float) -> bool:
    """[Patch] Check drawdown warning with debouncing."""
    if state.drawdown_pct >= threshold and not state.warned_drawdown:
        state.warned_drawdown = True
        return True
    return False


def should_warn_losses(state: CooldownState, threshold: int) -> bool:
    """[Patch] Check consecutive loss warning with debouncing."""
    if state.consecutive_losses >= threshold and not state.warned_losses:
        state.warned_losses = True
        return True
    return False
