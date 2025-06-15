"""Risk management helpers with kill switch support."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import logging
import math
from typing import Sequence

import pandas as pd
import numpy as np

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


def adjust_lot_recovery_mode(
    base_lot: float,
    consecutive_losses: int,
    loss_threshold: int,
    multiplier: float,
    min_lot: float,
) -> tuple[float, str]:
    """Increase lot size when losses exceed threshold."""
    if consecutive_losses >= loss_threshold:
        adjusted_lot = max(base_lot * multiplier, min_lot)
        if not math.isclose(adjusted_lot, base_lot):
            logging.info(
                "      (Recovery Mode Active) Losses: %s. Lot adjusted: %.2f -> %.2f",
                consecutive_losses,
                base_lot,
                adjusted_lot,
            )
        return adjusted_lot, "recovery"
    return base_lot, "normal"


def calculate_aggressive_lot(equity: float, max_lot: float, min_lot: float) -> float:
    """Return lot size using tiered equity levels."""
    if equity < 100:
        lot = 0.01
    elif equity < 500:
        lot = 0.05
    elif equity < 1000:
        lot = 0.10
    elif equity < 3000:
        lot = 0.30
    elif equity < 5000:
        lot = 0.50
    elif equity < 8000:
        lot = 1.00
    else:
        lot = 2.00
    final_lot = round(min(lot, max_lot), 2)
    return max(final_lot, min_lot)


def calculate_lot_size_fixed_risk(
    equity: float,
    risk_per_trade: float,
    sl_delta_price: float,
    point_value: float,
    min_lot: float,
    max_lot: float,
) -> float:
    """Lot size based on fixed fractional risk."""
    equity_num = pd.to_numeric(equity, errors="coerce")
    risk_num = pd.to_numeric(risk_per_trade, errors="coerce")
    sl_delta_num = pd.to_numeric(sl_delta_price, errors="coerce")
    if (
        pd.isna(equity_num)
        or np.isinf(equity_num)
        or equity_num <= 0
        or pd.isna(risk_num)
        or np.isinf(risk_num)
        or risk_num <= 0
        or pd.isna(sl_delta_num)
        or np.isinf(sl_delta_num)
        or sl_delta_num <= 1e-9
    ):
        return min_lot
    try:
        risk_amount_usd = equity_num * risk_num
        sl_points = sl_delta_num * 10.0
        risk_per_001_lot = sl_points * point_value
        if risk_per_001_lot <= 1e-9:
            return min_lot
        raw_lot_units = risk_amount_usd / risk_per_001_lot
        lot_size = raw_lot_units * 0.01
        lot_size = round(lot_size, 2)
        lot_size = max(min_lot, lot_size)
        lot_size = min(max_lot, lot_size)
        return lot_size
    except Exception:
        return min_lot


def adjust_lot_tp2_boost(
    trade_history: Sequence[str], base_lot: float, min_lot: float
) -> float:
    """Increase lot size slightly after consecutive TP trades."""
    boost_factor = 1.2
    streak_length = 2
    if len(trade_history) < streak_length:
        return base_lot
    full_trade_reasons = [str(reason) for reason in trade_history if not str(reason).startswith("Partial")]
    if len(full_trade_reasons) >= streak_length and all(t.upper() == "TP" for t in full_trade_reasons[-streak_length:]):
        boosted_lot = round(base_lot * boost_factor, 2)
        final_lot = max(boosted_lot, min_lot)
        if final_lot > base_lot:
            logging.info(
                "      (TP Boost) Last %s full trades were TP. Lot boosted: %.2f -> %.2f",
                streak_length,
                base_lot,
                final_lot,
            )
        return final_lot
    return base_lot


def calculate_lot_by_fund_mode(
    mm_mode: str,
    risk_pct: float,
    equity: float,
    atr_at_entry: float,
    sl_delta_price: float,
    min_lot: float,
    max_lot: float,
    point_value: float,
) -> float:
    """Determine base lot size based on money management mode."""
    base_lot = min_lot
    if mm_mode in ["conservative", "mirror"]:
        base_lot = calculate_lot_size_fixed_risk(
            equity,
            risk_pct,
            sl_delta_price,
            point_value,
            min_lot,
            max_lot,
        )
    elif mm_mode == "balanced":
        base_lot = calculate_aggressive_lot(equity, max_lot, min_lot)
    elif mm_mode == "high_freq":
        if equity < 100:
            base_lot = 0.01
        elif equity < 500:
            base_lot = 0.02
        elif equity < 1000:
            base_lot = 0.03
        else:
            base_lot = 0.05
    elif mm_mode == "spike_only":
        atr_threshold = 2.0
        atr_at_entry_num = pd.to_numeric(atr_at_entry, errors="coerce")
        if pd.notna(atr_at_entry_num) and atr_at_entry_num > atr_threshold:
            base_lot = calculate_aggressive_lot(equity, max_lot, min_lot)
        else:
            base_lot = calculate_lot_size_fixed_risk(
                equity,
                risk_pct,
                sl_delta_price,
                point_value,
                min_lot,
                max_lot,
            )
    else:
        base_lot = calculate_aggressive_lot(equity, max_lot, min_lot)
    final_lot = round(min(base_lot, max_lot), 2)
    return max(final_lot, min_lot)


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
    "adjust_lot_recovery_mode",
    "calculate_aggressive_lot",
    "calculate_lot_size_fixed_risk",
    "adjust_lot_tp2_boost",
    "calculate_lot_by_fund_mode",
    "RiskManager",
    "OrderStatus",
]
