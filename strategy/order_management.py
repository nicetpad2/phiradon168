"""Order management utilities with cooldown and kill switch logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict
import logging

from src.utils import load_settings, Settings


class OrderStatus(Enum):
    """Possible order placement outcomes."""

    OPEN = "OPEN"
    BLOCKED_COOLDOWN = "BLOCKED_COOLDOWN"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class OrderManager:
    """Manage order placement with cooldown and kill switch."""

    settings: Settings = field(default_factory=load_settings)

    def __post_init__(self) -> None:
        self._last_order_time: datetime | None = None
        self._current_drawdown: float = 0.0
        self._lock = Lock()

    def can_place_order(self, current_time: datetime) -> OrderStatus:
        """Check if an order can be placed at the given time."""
        with self._lock:
            if (
                self._last_order_time
                and (current_time - self._last_order_time)
                < timedelta(seconds=self.settings.cooldown_secs)
            ):
                logging.info(
                    "order_management: %s at %s",
                    OrderStatus.BLOCKED_COOLDOWN.value,
                    current_time.isoformat(),
                )
                return OrderStatus.BLOCKED_COOLDOWN
            if self._current_drawdown >= self.settings.kill_switch_pct:
                logging.info(
                    "order_management: %s at %s",
                    OrderStatus.KILL_SWITCH.value,
                    current_time.isoformat(),
                )
                return OrderStatus.KILL_SWITCH
            return OrderStatus.OPEN

    def place_order(self, order: Dict, current_time: datetime) -> OrderStatus:
        """Record the order if allowed and return the resulting status."""
        status = self.can_place_order(current_time)
        if status is OrderStatus.OPEN:
            with self._lock:
                self._last_order_time = current_time
        return status

    def update_drawdown(self, drawdown_pct: float) -> None:
        """Update current drawdown percentage for kill switch check."""
        with self._lock:
            self._current_drawdown = drawdown_pct


__all__ = ["OrderManager", "OrderStatus"]


def place_order(side: str, price: float, sl: float, tp: float, size: float) -> Dict:
    """Return an order dictionary with mandatory SL/TP."""
    return {
        "side": side,
        "entry_price": price,
        "sl_price": sl,
        "tp_price": tp,
        "size": size,
    }

__all__.append("place_order")


# compatibility wrapper

def create_order(side: str, price: float, sl: float, tp: float) -> Dict:
    """Create an order dictionary using legacy keys."""
    if sl is None or tp is None:
        raise ValueError("SL and TP must be provided")
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    return {
        "side": side,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
    }

__all__.append("create_order")
