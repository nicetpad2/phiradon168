"""Utility functions for cooldown logic."""

from __future__ import annotations

import logging
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


class CooldownManager:
    """Manage cooldown state within a trading session."""

    losses_count: int
    in_cooldown: bool
    cooldown_counter: int
    _cooldown_period: int
    _loss_threshold: int

    def __init__(self, loss_threshold: int = 3, cooldown_period: int = 5) -> None:
        if not isinstance(loss_threshold, int):
            raise TypeError("loss_threshold must be int")
        if not isinstance(cooldown_period, int):
            raise TypeError("cooldown_period must be int")
        self._loss_threshold = loss_threshold
        self._cooldown_period = cooldown_period
        self.losses_count = 0
        self.in_cooldown = False
        self.cooldown_counter = 0

    def record_loss(self) -> bool:
        """Record a loss and trigger cooldown if threshold reached."""

        if not isinstance(self.losses_count, int):
            raise TypeError("losses_count must be int")
        self.losses_count += 1
        if self.losses_count >= self._loss_threshold and not self.in_cooldown:
            self.in_cooldown = True
            self.cooldown_counter = self._cooldown_period
            logging.info(
                "Entering soft cooldown: losses_count=%d, threshold=%d",
                self.losses_count,
                self._loss_threshold,
            )
            return True
        return False

    def record_win(self) -> None:
        """Reset consecutive losses after a winning trade."""

        self.losses_count = 0

    def step(self) -> None:
        """Advance cooldown counter by one bar and exit if complete."""

        if self.in_cooldown:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            if self.cooldown_counter == 0:
                logging.info(
                    "Exiting soft cooldown after %d bars", self._cooldown_period
                )
                self.in_cooldown = False
                self.losses_count = 0

    def reset(self) -> None:
        """Reset all cooldown state."""

        self.losses_count = 0
        self.in_cooldown = False
        self.cooldown_counter = 0


def is_soft_cooldown_triggered(pnls: List[float], lookback: int = 15, loss_count: int = 2) -> tuple[bool, int]:
    """[Patch v5.0.24] Determine if soft cooldown should activate.

    If fewer than ``lookback`` trades exist, all available PnLs are considered.
    This ensures early trades can still trigger the cooldown.
    """

    if not isinstance(lookback, int):
        raise TypeError("lookback must be int")
    if not isinstance(loss_count, int):
        raise TypeError("loss_count must be int")

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
    if not isinstance(remaining_bars, int):
        raise TypeError("remaining_bars must be int")
    if not isinstance(step, int):
        raise TypeError("step must be int")
    if remaining_bars > 0:
        return max(0, remaining_bars - step)
    return 0


def update_losses(state: CooldownState, pnl: float) -> int:
    """[Patch] Update consecutive loss count based on PnL."""

    if not isinstance(pnl, (int, float)):
        raise TypeError("pnl must be numeric")
    if pnl < 0:
        state.consecutive_losses += 1
    else:
        state.consecutive_losses = 0
    return state.consecutive_losses


def update_drawdown(state: CooldownState, drawdown_pct: float) -> float:
    """[Patch] Update current drawdown percentage."""

    if not isinstance(drawdown_pct, (int, float)):
        raise TypeError("drawdown_pct must be numeric")
    state.drawdown_pct = float(drawdown_pct)
    return state.drawdown_pct


def should_enter_cooldown(state: CooldownState, loss_thresh: int, dd_thresh: float) -> bool:
    """[Patch] Determine if soft cooldown should start."""

    if not isinstance(loss_thresh, int):
        raise TypeError("loss_thresh must be int")
    if not isinstance(dd_thresh, (int, float)):
        raise TypeError("dd_thresh must be numeric")
    return state.consecutive_losses >= loss_thresh or state.drawdown_pct >= dd_thresh


def enter_cooldown(state: CooldownState, lookback: int) -> int:
    """[Patch] Begin a soft cooldown period."""

    if not isinstance(lookback, int):
        raise TypeError("lookback must be int")
    state.cooldown_bars_remaining = lookback
    return state.cooldown_bars_remaining


def should_warn_drawdown(state: CooldownState, threshold: float) -> bool:
    """[Patch] Check drawdown warning with debouncing."""

    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric")
    if state.drawdown_pct >= threshold and not state.warned_drawdown:
        state.warned_drawdown = True
        return True
    return False


def should_warn_losses(state: CooldownState, threshold: int) -> bool:
    """[Patch] Check consecutive loss warning with debouncing."""

    if not isinstance(threshold, int):
        raise TypeError("threshold must be int")
    if state.consecutive_losses >= threshold and not state.warned_losses:
        state.warned_losses = True
        return True
    return False
