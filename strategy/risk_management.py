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


__all__ = ["calculate_position_size", "RiskManager", "OrderStatus"]
