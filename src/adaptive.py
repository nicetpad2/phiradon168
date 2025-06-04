import json
import logging
from pathlib import Path


def adaptive_sl_tp(current_atr, avg_atr, base_sl=2.0, base_tp=1.8):
    """Return adaptive SL/TP multipliers based on ATR ratio."""
    try:
        current = float(current_atr)
        avg = float(avg_atr)
    except (TypeError, ValueError):
        return base_sl, base_tp
    if avg <= 1e-9:
        return base_sl, base_tp
    ratio = current / avg
    if ratio > 1.5:
        return base_sl * 1.2, base_tp * 1.2
    if ratio < 0.8:
        return base_sl * 0.8, base_tp * 0.8
    return base_sl, base_tp


def adaptive_risk(equity, peak_equity, base_risk=0.01, dd_threshold=0.1, min_risk=0.002):
    """Adjust risk per trade according to drawdown."""
    try:
        eq = float(equity)
        peak = float(peak_equity)
    except (TypeError, ValueError):
        return base_risk
    if peak <= 0:
        return base_risk
    dd = 1.0 - eq / peak
    if dd > dd_threshold:
        factor = max(1 - dd, min_risk / base_risk)
        return max(base_risk * factor, min_risk)
    return base_risk


def log_best_params(params, fold_index, output_dir):
    """Save best parameters for given fold as JSON."""
    path = Path(output_dir) / f"best_params_fold_{fold_index}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(params, fh, ensure_ascii=False, indent=2)
        logging.info(f"Saved best params to {path}")
        return path
    except Exception as e:  # pragma: no cover - logging only
        logging.error(f"Could not save best params: {e}")
        return None


def compute_kelly_position(current_winrate, win_loss_ratio):
    """Return Kelly fraction based on win rate and win/loss ratio."""
    try:
        p = float(current_winrate)
        b = float(win_loss_ratio)
        if b <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return 0.0

    kelly = p - (1 - p) / b
    return max(0.0, min(kelly, 1.0))


def compute_dynamic_lot(base_lot, drawdown_pct):
    """Reduce lot size based on drawdown percentage."""
    try:
        dd = float(drawdown_pct)
    except (TypeError, ValueError):
        return base_lot

    if dd > 0.10:
        return round(base_lot * 0.5, 2)
    if dd > 0.05:
        return round(base_lot * 0.75, 2)
    return base_lot


def compute_trailing_atr_stop(entry_price, current_price, atr_current, side, old_sl, atr_multiplier=1.5):
    """Return updated stop-loss price based on current ATR movement."""
    try:
        entry = float(entry_price)
        price = float(current_price)
        atr = float(atr_current)
        sl_old = float(old_sl)
    except (TypeError, ValueError):
        return old_sl

    if atr <= 0:
        return old_sl

    if side == 'BUY':
        profit = price - entry
        if profit >= 2 * atr:
            return max(sl_old, price - atr_multiplier * atr)
        if profit >= atr:
            return max(sl_old, entry)
    elif side == 'SELL':
        profit = entry - price
        if profit >= 2 * atr:
            return min(sl_old, price + atr_multiplier * atr)
        if profit >= atr:
            return min(sl_old, entry)

    return old_sl
