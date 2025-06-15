import logging
from typing import Tuple

import pandas as pd
import numpy as np


def apply_trend_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter entry signals using M15 Trend_Zone.

    * ถ้า Trend_Zone เป็น 'UP' ให้คงเฉพาะสัญญาณซื้อ (Entry_Long)
    * ถ้า Trend_Zone เป็น 'DOWN' ให้คงเฉพาะสัญญาณขาย (Entry_Short)
    * กรณีอื่น ๆ ปิดทั้งสองฝั่ง
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if "Trend_Zone" not in df.columns:
        return df.copy()
    result = df.copy()
    result["Entry_Long"] = np.where(result["Trend_Zone"] == "UP", result["Entry_Long"], 0).astype(int)
    result["Entry_Short"] = np.where(result["Trend_Zone"] == "DOWN", result["Entry_Short"], 0).astype(int)
    return result


def spike_guard_london(row: pd.Series, session: str, consecutive_losses: int) -> bool:
    """Spike guard filter for London session with debug reasons."""
    from src.strategy import ENABLE_SPIKE_GUARD

    if not ENABLE_SPIKE_GUARD:
        logging.debug("      (Spike Guard) Disabled via config.")
        return True
    if not isinstance(session, str) or "London" not in session:
        logging.debug("      (Spike Guard) Not London session - skipping.")
        return True

    spike_score_val = pd.to_numeric(getattr(row, "spike_score", np.nan), errors="coerce")
    if pd.notna(spike_score_val) and spike_score_val > 0.85:
        logging.debug(
            "      (Spike Guard Filtered) Reason: London Session & High Spike Score (%.2f > 0.85)",
            spike_score_val,
        )
        return False

    adx_val = pd.to_numeric(getattr(row, "ADX", np.nan), errors="coerce")
    wick_ratio_val = pd.to_numeric(getattr(row, "Wick_Ratio", np.nan), errors="coerce")
    vol_index_val = pd.to_numeric(getattr(row, "Volatility_Index", np.nan), errors="coerce")
    candle_body_val = pd.to_numeric(getattr(row, "Candle_Body", np.nan), errors="coerce")
    candle_range_val = pd.to_numeric(getattr(row, "Candle_Range", np.nan), errors="coerce")
    gain_val = pd.to_numeric(getattr(row, "Gain", np.nan), errors="coerce")
    atr_val = pd.to_numeric(getattr(row, "ATR_14", np.nan), errors="coerce")

    if any(pd.isna(v) for v in [adx_val, wick_ratio_val, vol_index_val, candle_body_val, candle_range_val, gain_val, atr_val]):
        logging.debug("      (Spike Guard) Missing values - skip filter.")
        return True

    safe_candle_range_val = max(candle_range_val, 1e-9)

    if adx_val < 20 and wick_ratio_val > 0.7 and vol_index_val < 0.8:
        logging.debug(
            "      (Spike Guard Filtered) Reason: Low ADX(%.1f), High Wick(%.2f), Low Vol(%.2f)",
            adx_val,
            wick_ratio_val,
            vol_index_val,
        )
        return False

    try:
        body_ratio = candle_body_val / safe_candle_range_val
        if body_ratio < 0.07:
            logging.debug("      (Spike Guard Filtered) Reason: Low Body Ratio(%.3f)", body_ratio)
            return False
    except ZeroDivisionError:
        logging.warning("      (Spike Guard) ZeroDivisionError calculating body_ratio.")
        return False

    if gain_val > 3 and atr_val > 4 and (candle_body_val / safe_candle_range_val) > 0.3:
        logging.debug("      (Spike Guard Allowed) Reason: Strong directional move override.")
        return True

    logging.debug("      (Spike Guard) Passed all checks.")
    return True


def is_mtf_trend_confirmed(m15_trend: str | None, side: str) -> bool:
    """Validate entry direction using M15 trend zone."""
    from src.strategy import M15_TREND_ALLOWED

    trend = str(m15_trend).upper() if isinstance(m15_trend, str) else "NEUTRAL"
    if side == "BUY" and trend not in M15_TREND_ALLOWED:
        return False
    if side == "SELL" and trend not in ["DOWN", "NEUTRAL"]:
        return False
    return True


def passes_volatility_filter(vol_index: float, min_ratio: float = 1.0) -> bool:
    """Return True if Volatility_Index >= min_ratio."""
    vol_val = pd.to_numeric(vol_index, errors="coerce")
    if pd.isna(vol_val):
        return False
    return vol_val >= min_ratio


def is_entry_allowed(
    row: pd.Series,
    session: str,
    consecutive_losses: int,
    side: str,
    m15_trend: str | None = None,
    signal_score_threshold: float | None = None,
) -> Tuple[bool, str]:
    """Checks if entry is allowed based on filters with debug logging."""
    from src.strategy import MIN_SIGNAL_SCORE_ENTRY

    if signal_score_threshold is None:
        signal_score_threshold = MIN_SIGNAL_SCORE_ENTRY

    if not spike_guard_london(row, session, consecutive_losses):
        logging.debug("      Entry blocked by Spike Guard.")
        return False, "SPIKE_GUARD_LONDON"

    if not is_mtf_trend_confirmed(m15_trend, side):
        logging.debug("      Entry blocked by M15 Trend filter.")
        return False, f"M15_TREND_{str(m15_trend).upper()}"

    vol_index_val = pd.to_numeric(getattr(row, "Volatility_Index", np.nan), errors="coerce")
    if not passes_volatility_filter(vol_index_val):
        logging.debug("      Entry blocked by Low Volatility (%s)", vol_index_val)
        return False, f"LOW_VOLATILITY({vol_index_val})"

    signal_score = pd.to_numeric(getattr(row, "Signal_Score", np.nan), errors="coerce")
    if pd.isna(signal_score):
        logging.debug("      Entry blocked: Invalid Signal Score (NaN)")
        return False, "INVALID_SIGNAL_SCORE (NaN)"
    if abs(signal_score) < signal_score_threshold:
        logging.debug(
            "      Entry blocked: Low Signal Score %.2f < %.2f",
            signal_score,
            signal_score_threshold,
        )
        return False, f"LOW_SIGNAL_SCORE ({signal_score:.2f}<{signal_score_threshold})"

    logging.debug("      Entry allowed by filters.")
    return True, "ALLOWED"


__all__ = [
    "apply_trend_filter",
    "spike_guard_london",
    "is_mtf_trend_confirmed",
    "passes_volatility_filter",
    "is_entry_allowed",
]
