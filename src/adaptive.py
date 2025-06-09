import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import warnings
try:
    import torch
except OSError as e:  # pragma: no cover - optional GPU
    warnings.warn(
        f"CUDA libraries not found ({e}), defaulting to CPU-only mode"
    )

    class _DummyCuda:
        def is_available(self) -> bool:
            return False

    class _DummyTorch:
        cuda = _DummyCuda()

    torch = _DummyTorch()

import pandas as pd
from src import features

logger = logging.getLogger(__name__)


def adaptive_sl_tp(
    current_atr: float,
    avg_atr: float,
    base_sl: float = 2.0,
    base_tp: float = 1.8,
) -> Tuple[float, float]:
    """Return adaptive SL/TP multipliers based on ATR ratio."""
    logger.debug(
        "Calculating adaptive_sl_tp with current_atr=%s avg_atr=%s", current_atr, avg_atr
    )
    try:
        current = float(current_atr)
        avg = float(avg_atr)
    except (TypeError, ValueError):
        logger.warning("Received invalid ATR inputs: %s, %s", current_atr, avg_atr)
        return base_sl, base_tp
    if avg <= 1e-9:
        logger.warning("Average ATR is non-positive: %s", avg)
        return base_sl, base_tp
    ratio = current / avg
    if ratio > 1.5:
        return base_sl * 1.2, base_tp * 1.2
    if ratio < 0.8:
        return base_sl * 0.8, base_tp * 0.8
    return base_sl, base_tp


def adaptive_risk(
    equity: float,
    peak_equity: float,
    base_risk: float = 0.01,
    dd_threshold: float = 0.1,
    min_risk: float = 0.002,
) -> float:
    """Adjust risk per trade according to drawdown."""
    logger.debug(
        "Calculating adaptive_risk with equity=%s peak_equity=%s", equity, peak_equity
    )
    try:
        eq = float(equity)
        peak = float(peak_equity)
    except (TypeError, ValueError):
        logger.warning("Invalid equity inputs: %s, %s", equity, peak_equity)
        return base_risk
    if peak <= 0:
        logger.warning("Peak equity is non-positive: %s", peak_equity)
        return base_risk
    dd = 1.0 - eq / peak
    if dd > dd_threshold:
        factor = max(1 - dd, min_risk / base_risk)
        return max(base_risk * factor, min_risk)
    return base_risk


def log_best_params(params: dict, fold_index: int, output_dir: str) -> Optional[Path]:
    """Save best parameters for given fold as JSON."""
    path = Path(output_dir) / f"best_params_fold_{fold_index}.json"
    logger.debug("Saving best params for fold %s to %s", fold_index, path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(params, fh, ensure_ascii=False, indent=2)
        logger.info("Saved best params to %s", path)
        return path
    except Exception as e:  # pragma: no cover - logging only
        logger.error("Could not save best params: %s", e)
        return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value using :func:`features.atr`."""
    logger.debug("Calculating ATR for period=%s", period)
    if not isinstance(df, pd.DataFrame):
        logger.warning("calculate_atr received invalid dataframe: %s", type(df))
        return float("nan")
    try:
        atr_df = features.atr(df[["High", "Low", "Close"]].copy(), period=period)
        col = f"ATR_{period}"
        if col in atr_df.columns:
            val = pd.to_numeric(atr_df[col].iloc[-1], errors="coerce")
            return float(val)
    except Exception as e:  # pragma: no cover - logging only
        logger.error("calculate_atr failed: %s", e)
    return float("nan")


def atr_position_size(
    equity: float,
    atr: float,
    risk_pct: float = 0.01,
    atr_mult: float = 1.5,
    pip_value: float = 0.1,
    min_lot: float = 0.01,
    max_lot: float = 5.0,
) -> Tuple[float, float]:
    """Calculate lot size and SL distance based on ATR."""
    logger.debug(
        "Calculating atr_position_size for equity=%s atr=%s", equity, atr
    )
    try:
        eq = float(equity)
        atr_val = float(atr)
    except (TypeError, ValueError):
        logger.warning("Invalid inputs for position size: %s, %s", equity, atr)
        return min_lot, float("nan")
    if eq <= 0 or atr_val <= 0 or pip_value <= 0:
        logger.warning(
            "Non-positive values detected eq=%s atr=%s pip_value=%s", eq, atr_val, pip_value
        )
        return min_lot, float("nan")

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


def compute_kelly_position(current_winrate: float, win_loss_ratio: float) -> float:
    """Return Kelly position size percentage."""
    logger.debug(
        "Computing Kelly position for winrate=%s ratio=%s", current_winrate, win_loss_ratio
    )
    try:
        p = float(current_winrate)
        b = float(win_loss_ratio)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid Kelly inputs winrate=%s ratio=%s", current_winrate, win_loss_ratio
        )
        return 0.0
    if b <= 0:
        logger.warning("win_loss_ratio <= 0 อาจผิดพลาดในข้อมูล")
        return 0.0

    kelly = (p - (1 - p) / b) * 100
    return max(0.0, kelly)


def compute_dynamic_lot(base_lot: float, drawdown_pct: float) -> float:
    """Reduce lot size based on drawdown percentage."""
    logger.debug("compute_dynamic_lot base_lot=%s drawdown_pct=%s", base_lot, drawdown_pct)
    try:
        dd = float(drawdown_pct)
    except (TypeError, ValueError):
        logger.warning("Invalid drawdown_pct: %s", drawdown_pct)
        return base_lot

    if dd > 0.10:
        return round(base_lot * 0.5, 2)
    if dd > 0.05:
        return round(base_lot * 0.75, 2)
    return base_lot


def compute_trailing_atr_stop(
    entry_price: float,
    current_price: float,
    atr_current: float,
    side: str,
    old_sl: float,
    atr_multiplier: float = 1.5,
) -> float:
    """Return updated stop-loss price based on current ATR movement."""
    logger.debug(
        "compute_trailing_atr_stop entry=%s price=%s atr=%s side=%s old_sl=%s",
        entry_price,
        current_price,
        atr_current,
        side,
        old_sl,
    )
    try:
        entry = float(entry_price)
        price = float(current_price)
        atr = float(atr_current)
        sl_old = float(old_sl)
    except (TypeError, ValueError):
        logger.warning("Invalid trailing stop inputs")
        return old_sl

    if atr <= 0:
        logger.warning("ATR must be positive for trailing stop")
        return old_sl

    side = side.upper()
    if side == "BUY":
        profit = price - entry
        if profit >= 2 * atr:
            return max(sl_old, price - atr_multiplier * atr)
        if profit >= atr:
            return max(sl_old, entry)
    elif side == "SELL":
        profit = entry - price
        if profit >= 2 * atr:
            return min(sl_old, price + atr_multiplier * atr)
        if profit >= atr:
            return min(sl_old, entry)

    return old_sl

def volatility_adjusted_lot_size(
    equity: float,
    atr_value: float,
    sl_multiplier: float = 1.5,
    pip_value: float = 0.1,
    risk_pct: float = 0.01,
    min_lot: float = 0.01,
    max_lot: float = 5.0,
) -> Tuple[float, float]:
    """[Patch] Calculate lot size based on ATR volatility."""
    try:
        equity = float(equity)
        atr_value = float(atr_value)
    except (TypeError, ValueError):
        logger.warning("Invalid inputs for volatility_adjusted_lot_size")
        return min_lot, float("nan")
    if equity <= 0 or atr_value <= 0 or pip_value <= 0:
        logger.warning("Non-positive inputs for volatility_adjusted_lot_size")
        return min_lot, float("nan")
    sl_pips = atr_value * sl_multiplier * 10.0
    risk_amount = equity * risk_pct
    risk_per_lot = sl_pips * pip_value
    if risk_per_lot <= 1e-9:
        return min_lot, atr_value * sl_multiplier
    lot = round(risk_amount / risk_per_lot, 2)
    lot = max(min_lot, min(lot, max_lot))
    return lot, atr_value * sl_multiplier


def dynamic_risk_adjustment(
    fold_returns: list[float],
    base_risk: float = 0.01,
    loss_cutoff: float = -0.05,
    win_cutoff: float = 0.05,
) -> float:
    """[Patch] Adjust risk based on consecutive fold performance."""
    if not fold_returns:
        return base_risk
    last_three = fold_returns[-3:]
    last_two = fold_returns[-2:]
    if len(last_three) >= 2 and all(r <= loss_cutoff for r in last_three[-2:]):
        return base_risk * 0.5
    if len(last_three) == 3 and all(r <= loss_cutoff for r in last_three):
        return base_risk * 0.5  # pragma: no cover
    if len(last_two) == 2 and all(r >= win_cutoff for r in last_two):
        return base_risk * 1.5
    return base_risk


def check_portfolio_stop(drawdown_pct: float, threshold: float = 0.10) -> bool:
    """[Patch] Return True if trading should be suspended."""
    try:
        dd = float(drawdown_pct)
    except (TypeError, ValueError):
        logger.warning("Invalid drawdown_pct for portfolio stop")
        return False
    return dd >= threshold


def calculate_dynamic_sl_tp(
    atr: float,
    win_rate: float,
    sl_min_pips: float = 20.0,
    sl_max_pips: float = 30.0,
    atr_multiplier: float = 1.5,
) -> Tuple[float, float]:
    """Return SL/TP distances based on ATR and win rate.

    - SL ใช้ค่าระหว่าง ``sl_min_pips`` ถึง ``sl_max_pips``
      โดยอิงจาก ``atr * atr_multiplier`` ในหน่วยราคา
    - TP จะปรับตาม ``win_rate``:
      * ต่ำกว่า 40% ⇒ TP = 3×SL
      * สูงกว่า 50% ⇒ TP = 1.5×SL
      * อื่น ๆ ⇒ TP = 2×SL
    """

    try:
        atr_val = float(atr)
        wr = float(win_rate)
    except (TypeError, ValueError):
        logger.warning("Invalid inputs for calculate_dynamic_sl_tp")
        return float("nan"), float("nan")

    sl_by_atr = atr_val * atr_multiplier
    sl_min = sl_min_pips / 10.0
    sl_max = sl_max_pips / 10.0
    sl = max(sl_min, min(sl_by_atr, sl_max))

    if wr < 0.40:
        tp_mult = 3.0
    elif wr > 0.50:
        tp_mult = 1.5
    else:
        tp_mult = 2.0

    tp = sl * tp_mult
    return sl, tp


def update_signal_threshold(current_score: float, params) -> float:
    """[Patch] Adjust ``signal_score_threshold`` based on ``current_score``.

    หากมีการปรับค่าจะบันทึก Log รูปแบบ
    ``[Adaptive] Threshold changed | Fold=<…> | Profile=<…> | Old=<…> -> New=<…> | Current Score=<…>``
    """

    try:
        score = float(current_score)
    except (TypeError, ValueError):
        logger.warning("Invalid current_score for update_signal_threshold")
        return params.signal_score_threshold

    some_condition = score > 0.8
    other_condition = score < 0.2

    if some_condition:
        old_th = params.signal_score_threshold
        new_th = 0.50
        params.signal_score_threshold = new_th
        logger.info(
            "[Adaptive] Threshold changed | Fold=%s | Profile=%s | Old=%s -> New=%s | Current Score=%.2f",
            getattr(params, "current_fold", "N/A"),
            getattr(params, "profile_name", "N/A"),
            old_th,
            new_th,
            score,
        )
    elif other_condition:
        old_th = params.signal_score_threshold
        new_th = 0.25
        params.signal_score_threshold = new_th
        logger.info(
            "[Adaptive] Threshold changed | Fold=%s | Profile=%s | Old=%s -> New=%s | Current Score=%.2f",
            getattr(params, "current_fold", "N/A"),
            getattr(params, "profile_name", "N/A"),
            old_th,
            new_th,
            score,
        )

    return params.signal_score_threshold
