"""Trade utility helper functions extracted from strategy."""

import logging
import math
import pandas as pd
import numpy as np


def dynamic_tp2_multiplier(current_atr, avg_atr, base=None):
    from src import strategy as _s
    if base is None:
        base = getattr(_s, "BASE_TP_MULTIPLIER", 1.8)
    current_atr_num = pd.to_numeric(current_atr, errors="coerce")
    avg_atr_num = pd.to_numeric(avg_atr, errors="coerce")
    if (
        pd.isna(current_atr_num)
        or pd.isna(avg_atr_num)
        or np.isinf(current_atr_num)
        or np.isinf(avg_atr_num)
        or avg_atr_num < 1e-9
    ):
        return base
    try:
        ratio = current_atr_num / avg_atr_num
        high_vol_ratio = getattr(_s, "ADAPTIVE_TSL_HIGH_VOL_RATIO", 1.8)
        high_vol_adjust = 0.6
        mid_vol_ratio = 1.2
        mid_vol_adjust = 0.3
        if ratio >= high_vol_ratio:
            return base + high_vol_adjust
        if ratio >= mid_vol_ratio:
            return base + mid_vol_adjust
        return base
    except Exception:
        return base


def get_adaptive_tsl_step(current_atr, avg_atr, default_step=None):
    from src import strategy as _s
    if default_step is None:
        default_step = getattr(_s, "ADAPTIVE_TSL_DEFAULT_STEP_R", 0.5)
    high_vol_ratio = getattr(_s, "ADAPTIVE_TSL_HIGH_VOL_RATIO", 1.8)
    high_vol_step = getattr(_s, "ADAPTIVE_TSL_HIGH_VOL_STEP_R", 1.0)
    low_vol_ratio = getattr(_s, "ADAPTIVE_TSL_LOW_VOL_RATIO", 0.75)
    low_vol_step = getattr(_s, "ADAPTIVE_TSL_LOW_VOL_STEP_R", 0.3)

    current_atr_num = pd.to_numeric(current_atr, errors="coerce")
    avg_atr_num = pd.to_numeric(avg_atr, errors="coerce")
    if (
        pd.isna(current_atr_num)
        or pd.isna(avg_atr_num)
        or np.isinf(current_atr_num)
        or np.isinf(avg_atr_num)
        or avg_atr_num < 1e-9
    ):
        return default_step
    try:
        ratio = current_atr_num / avg_atr_num
        if ratio > high_vol_ratio:
            return high_vol_step
        if ratio < low_vol_ratio:
            return low_vol_step
        return default_step
    except Exception:
        return default_step


def get_dynamic_signal_score_entry(df, window=1000, quantile=0.7, min_val=0.5, max_val=3.0):
    if df is None or "Signal_Score" not in df.columns or len(df) == 0:
        return min_val
    scores = df["Signal_Score"].dropna().astype(float)
    recent_scores = scores.iloc[-window:]
    if recent_scores.empty:
        return min_val
    val = recent_scores.quantile(quantile)
    val = max(min_val, min(val, max_val))
    return float(val)


def get_dynamic_signal_score_thresholds(series: pd.Series, window: int = 1000, quantile: float = 0.7,
                                        min_val: float = 0.5, max_val: float = 3.0) -> np.ndarray:
    scores = pd.to_numeric(series, errors="coerce")
    thresh = scores.rolling(window=window, min_periods=1).quantile(quantile)
    thresh = thresh.clip(lower=min_val, upper=max_val).fillna(min_val)
    return thresh.to_numpy()


def get_dynamic_signal_score_thresholds_atr(
    signal_series: pd.Series,
    atr_series: pd.Series,
    atr_avg_series: pd.Series,
    window: int = 1000,
    quantile: float = 0.7,
    slope: float = 0.2,
    min_val: float = 0.5,
    max_val: float = 3.0,
) -> np.ndarray:
    """Return dynamic signal score thresholds adjusted by ATR ratio."""
    scores = pd.to_numeric(signal_series, errors="coerce")
    base = scores.rolling(window=window, min_periods=1).quantile(quantile)
    atr = pd.to_numeric(atr_series, errors="coerce")
    atr_avg = pd.to_numeric(atr_avg_series, errors="coerce")
    ratio = atr / atr_avg.replace(0, np.nan)
    dynamic = slope * (ratio - 1.0)
    thresh = (base + dynamic).clip(lower=min_val, upper=max_val).fillna(min_val)
    return thresh.to_numpy()


__all__ = [
    "dynamic_tp2_multiplier",
    "get_adaptive_tsl_step",
    "get_dynamic_signal_score_entry",
    "get_dynamic_signal_score_thresholds",
    "get_dynamic_signal_score_thresholds_atr",
]
