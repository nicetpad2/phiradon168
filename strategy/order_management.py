"""Order management utilities with cooldown and kill switch logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict
import logging
import math
import pandas as pd

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


def adjust_sl_tp_oms(
    entry_price: float,
    sl_price: float,
    tp_price: float,
    atr: float,
    side: str,
    margin_pips: float,
    max_pips: float,
) -> tuple[float, float]:
    """Validate SL/TP distance and auto-adjust if outside allowed range."""
    if any(pd.isna(v) for v in [entry_price, sl_price, tp_price, atr]):
        return sl_price, tp_price

    sl_dist = abs(entry_price - sl_price) * 10.0
    tp_dist = abs(tp_price - entry_price) * 10.0

    if sl_dist < margin_pips:
        adj = atr if pd.notna(atr) and atr > 1e-9 else margin_pips / 10.0
        sl_price = entry_price - adj if side == "BUY" else entry_price + adj
        logging.info("[OMS_Guardian] Adjust SL to margin level: %.5f", sl_price)

    if sl_dist > max_pips:
        sl_price = entry_price - atr if side == "BUY" else entry_price + atr
        logging.info(
            "[OMS_Guardian] SL distance too wide. Adjusted to %.5f", sl_price
        )

    if tp_dist > max_pips:
        tp_price = entry_price + atr if side == "BUY" else entry_price - atr
        logging.info(
            "[OMS_Guardian] TP distance too wide. Adjusted to %.5f", tp_price
        )

    return sl_price, tp_price


def update_breakeven_half_tp(
    order: Dict,
    current_high: float,
    current_low: float,
    now: pd.Timestamp,
    entry_buffer: float = 0.0001,
) -> tuple[Dict, bool]:
    """Move SL to breakeven when price moves halfway to TP1."""
    if order.get("be_triggered", False):
        return order, False

    side = order.get("side")
    entry = pd.to_numeric(order.get("entry_price"), errors="coerce")
    tp1 = pd.to_numeric(order.get("tp1_price"), errors="coerce")
    sl = pd.to_numeric(order.get("sl_price"), errors="coerce")

    if any(pd.isna(v) for v in [side, entry, tp1, sl]):
        return order, False

    trigger = entry + 0.5 * (tp1 - entry) if side == "BUY" else entry - 0.5 * (
        entry - tp1
    )
    hit = (side == "BUY" and current_high >= trigger) or (
        side == "SELL" and current_low <= trigger
    )

    if hit:
        new_sl = entry + entry_buffer if side == "BUY" else entry - entry_buffer
        if not math.isclose(new_sl, sl, rel_tol=1e-9, abs_tol=1e-9):
            order["sl_price"] = new_sl
            order["be_triggered"] = True
            order["be_triggered_time"] = now
            logging.info("Move to Breakeven at price %.5f", new_sl)
            return order, True

    return order, False


__all__ += ["adjust_sl_tp_oms", "update_breakeven_half_tp"]
