import json
import logging
from pathlib import Path
import pandas as pd
from src import features


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


def calculate_atr(df, period=14):
    """Return the latest ATR value using features.atr."""
    if not isinstance(df, pd.DataFrame):
        return float('nan')
    try:
        atr_df = features.atr(df[['High', 'Low', 'Close']].copy(), period=period)
        col = f"ATR_{period}"
        if col in atr_df.columns:
            val = pd.to_numeric(atr_df[col].iloc[-1], errors='coerce')
            return float(val)
    except Exception as e:  # pragma: no cover - logging only
        logging.error(f"calculate_atr failed: {e}")
    return float('nan')


def atr_position_size(
    equity,
    atr,
    risk_pct=0.01,
    atr_mult=1.5,
    pip_value=0.1,
    min_lot=0.01,
    max_lot=5.0,
):
    """Calculate lot size and SL distance based on ATR."""
    try:
        eq = float(equity)
        atr_val = float(atr)
    except (TypeError, ValueError):
        return min_lot, float('nan')
    if eq <= 0 or atr_val <= 0 or pip_value <= 0:
        return min_lot, float('nan')

    sl_delta_price = atr_val * atr_mult
    risk_amount_usd = eq * risk_pct
    sl_pips = sl_delta_price * 10.0
    risk_per_001 = sl_pips * pip_value
    if risk_per_001 <= 1e-9:
        return min_lot, sl_delta_price

    lot_units = risk_amount_usd / risk_per_001
    lot = round(lot_units * 0.01, 2)
    lot = max(min_lot, min(lot, max_lot))
    return lot, sl_delta_price


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
