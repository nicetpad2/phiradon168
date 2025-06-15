import logging
from typing import Tuple, Dict
from src.adaptive import (
    compute_trailing_atr_stop,
    check_portfolio_stop,
    volatility_adjusted_lot_size,
)

logger = logging.getLogger(__name__)


def atr_sl_tp(
    entry_price: float,
    atr: float,
    side: str,
    sl_mult: float = 2.0,
    tp_mult: float = 2.0,
) -> Tuple[float, float]:
    """Return stop-loss and take-profit prices based on ATR.

    Parameters
    ----------
    entry_price : float
        Price at entry.
    atr : float
        Average True Range value.
    side : str
        Trade side ``BUY`` or ``SELL``.
    sl_mult : float, optional
        ATR multiplier for stop loss. Defaults to ``2.0``.
    tp_mult : float, optional
        ATR multiplier for take profit. Defaults to ``2.0``.
    """
    try:
        price = float(entry_price)
        atr_val = float(atr)
    except (TypeError, ValueError):
        logger.warning("Invalid inputs for atr_sl_tp")
        return float("nan"), float("nan")
    if atr_val <= 0:
        logger.warning("ATR must be positive for atr_sl_tp")
        return float("nan"), float("nan")

    side_u = str(side).upper()
    sl_delta = atr_val * max(sl_mult, 0.0)
    tp_delta = atr_val * max(tp_mult, 0.0)

    if side_u == "BUY":
        sl = price - sl_delta
        tp = price + tp_delta
    elif side_u == "SELL":
        sl = price + sl_delta
        tp = price - tp_delta
    else:
        logger.warning("Unknown side for atr_sl_tp: %s", side)
        return float("nan"), float("nan")
    return sl, tp


def update_be_trailing(order: Dict, current_price: float, atr: float, side: str, trailing_mult: float = 1.5) -> Dict:
    """Apply breakeven and trailing stop adjustments to order."""
    if order is None:
        return {}
    try:
        entry = float(order.get("entry_price"))
        sl = float(order.get("sl_price"))
        price = float(current_price)
        atr_val = float(atr)
    except (TypeError, ValueError):
        logger.warning("Invalid inputs for update_be_trailing")
        return order

    side_u = str(side).upper()
    if not order.get("be_triggered", False):
        if side_u == "BUY" and price - entry >= atr_val:
            order["sl_price"] = entry
            order["be_triggered"] = True
        elif side_u == "SELL" and entry - price >= atr_val:
            order["sl_price"] = entry
            order["be_triggered"] = True
    else:
        new_sl = compute_trailing_atr_stop(entry, price, atr_val, side_u, sl, trailing_mult)
        if side_u == "BUY":
            order["sl_price"] = max(sl, new_sl)
        else:
            order["sl_price"] = min(sl, new_sl)
    return order


def adaptive_position_size(equity: float, atr: float, risk_pct: float = 0.01) -> float:
    """[Patch v6.8.5] Return position size using volatility-adjusted calculation."""
    # [Patch] leverage volatility_adjusted_lot_size for dynamic sizing
    lot, _ = volatility_adjusted_lot_size(equity, atr, risk_pct=risk_pct)
    return lot


def portfolio_hard_stop(peak_equity: float, current_equity: float, threshold: float = 0.10) -> bool:
    """Return True if portfolio drawdown exceeds threshold."""
    try:
        peak = float(peak_equity)
        eq = float(current_equity)
    except (TypeError, ValueError):
        logger.warning("Invalid equity values for portfolio_hard_stop")
        return False
    if peak <= 0:
        return False
    dd = (peak - eq) / peak
    return check_portfolio_stop(dd, threshold)
