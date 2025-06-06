"""Risk management helpers with kill switch support."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import logging

from src.utils import load_settings, Settings


class OrderStatus(Enum):
    OPEN = "OPEN"
    BLOCKED_COOLDOWN = "BLOCKED_COOLDOWN"
    KILL_SWITCH = "KILL_SWITCH"


def calculate_position_size(balance: float, risk_pct: float, sl_distance: float) -> float:
    """Compute lot size based on risk percent and stop loss distance."""
    if sl_distance <= 0 or balance <= 0:
        raise ValueError("invalid inputs")
    risk_amount = balance * risk_pct
    size = risk_amount / sl_distance
    return max(size, 0.0)


# [Patch v5.8.7] Add utility functions for dynamic risk sizing
def compute_lot_size(
    equity: float,
    risk_pct: float,
    sl_pips: float,
    pip_value: float = 0.1,
    min_lot: float = 0.01,
) -> float:
    """Return lot size using fixed risk per trade."""
    if equity <= 0 or risk_pct <= 0 or sl_pips <= 0 or pip_value <= 0:
        raise ValueError("invalid inputs for compute_lot_size")
    risk_amount = equity * risk_pct
    risk_per_001 = sl_pips * pip_value
    lot = risk_amount / risk_per_001 * 0.01
    return max(lot, min_lot)


def adjust_risk_by_equity(
    equity: float,
    base_risk_pct: float = 0.01,
    low_equity_threshold: float = 500.0,
) -> float:
    """Reduce risk percentage when equity drops below threshold."""
    if equity < 0 or base_risk_pct <= 0:
        raise ValueError("invalid inputs for adjust_risk_by_equity")
    return 0.005 if equity < low_equity_threshold else base_risk_pct


def dynamic_position_size(
    base_lot: float,
    atr_current: float,
    atr_avg: float,
    high_ratio: float = 1.5,
    low_ratio: float = 0.75,
    fixed_lot: float = 0.05,
) -> float:
    """Adjust lot size based on volatility."""
    if base_lot <= 0 or atr_current <= 0 or atr_avg <= 0:
        raise ValueError("invalid inputs for dynamic_position_size")
    ratio = atr_current / atr_avg
    if ratio > high_ratio:
        return round(base_lot * 0.8, 2)
    if ratio < low_ratio:
        return fixed_lot
    return base_lot


def check_max_daily_drawdown(
    start_equity: float,
    current_equity: float,
    threshold_pct: float = 0.02,
) -> bool:
    """Return True if drawdown from day's start exceeds threshold."""
    if start_equity <= 0 or current_equity < 0 or threshold_pct <= 0:
        raise ValueError("invalid inputs for check_max_daily_drawdown")
    dd = (start_equity - current_equity) / start_equity
    return dd >= threshold_pct


def check_trailing_equity_stop(
    peak_equity: float,
    current_equity: float,
    threshold_pct: float = 0.05,
) -> bool:
    """Return True if equity falls from peak beyond threshold."""
    if peak_equity <= 0 or current_equity < 0 or threshold_pct <= 0:
        raise ValueError("invalid inputs for check_trailing_equity_stop")
    drop = (peak_equity - current_equity) / peak_equity
    return drop >= threshold_pct


def can_open_trade(open_trades: int, max_open: int = 2) -> bool:
    """Return True if a new trade can be opened."""
    if open_trades < 0 or max_open <= 0:
        raise ValueError("invalid inputs for can_open_trade")
    return open_trades < max_open


# [Patch v5.8.8] Hard cut-off checker
def should_hard_cutoff(
    daily_drawdown_pct: float,
    consecutive_losses: int,
    dd_threshold: float = 0.03,
    loss_threshold: int = 5,
) -> bool:
    """Return True if trading should stop for the day."""
    if not isinstance(daily_drawdown_pct, (int, float)):
        raise TypeError("daily_drawdown_pct must be numeric")
    if not isinstance(consecutive_losses, int):
        raise TypeError("consecutive_losses must be int")
    if daily_drawdown_pct >= dd_threshold or consecutive_losses >= loss_threshold:
        return True
    return False


@dataclass
class RiskManager:
    """Track drawdown and enforce kill switch."""

    settings: Settings = field(default_factory=load_settings)

    def __post_init__(self) -> None:
        self._current_drawdown = 0.0
        self._lock = Lock()

    def update_drawdown(self, drawdown_pct: float) -> None:
        with self._lock:
            self._current_drawdown = drawdown_pct

    def check_kill_switch(self) -> OrderStatus:
        with self._lock:
            if self._current_drawdown >= self.settings.kill_switch_pct:
                logging.info("risk_management: %s", OrderStatus.KILL_SWITCH.value)
                return OrderStatus.KILL_SWITCH
            return OrderStatus.OPEN


__all__ = [
    "calculate_position_size",
    "compute_lot_size",
    "adjust_risk_by_equity",
    "dynamic_position_size",
    "check_max_daily_drawdown",
    "check_trailing_equity_stop",
    "can_open_trade",
    "should_hard_cutoff",
    "RiskManager",
    "OrderStatus",
]
