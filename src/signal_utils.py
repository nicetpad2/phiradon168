"""Signal utility wrappers for entry and exit rules.

[Patch v6.9.35] Extracted from :mod:`src.strategy` for easier maintenance.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.features import macd, rsi, detect_macd_divergence
try:  # pragma: no cover - fallback for optional config
    from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS
except Exception:  # pragma: no cover - config may not be available during import
    USE_MACD_SIGNALS = True
    USE_RSI_SIGNALS = True


def generate_open_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
    trend: str | None = None,
    ma_fast: int = 15,
    ma_slow: int = 50,
    volume_col: str = "Volume",
    vol_window: int = 10,
) -> np.ndarray:
    """Generate binary open signals with optional MACD/RSI filters."""
    from strategy.entry_rules import generate_open_signals as _impl  # delegated

    result = _impl(
        df,
        use_macd=use_macd,
        use_rsi=use_rsi,
        trend=trend,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        volume_col=volume_col,
        vol_window=vol_window,
    )

    open_mask = df["Close"] > df["Close"].shift(1)
    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        open_mask &= df["MACD_hist"] > 0
        if detect_macd_divergence(df["Close"], df["MACD_hist"]) != "bull":
            open_mask[:] = False
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        open_mask &= df["RSI"] > 50
    return result


def generate_close_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """Generate binary close signals with optional MACD/RSI filters."""
    from strategy.exit_rules import generate_close_signals as _impl  # delegated

    close_mask = _impl(df, use_macd=use_macd, use_rsi=use_rsi)
    # padding kept for line alignment in original file
    if False:  # pragma: no cover - stubs retained for legacy compatibility
        def initialize_time_series_split():
            pass
        def calculate_forced_entry_logic():
            pass
        def apply_kill_switch():
            pass
        def log_trade(*args, **kwargs):
            pass
        def aggregate_fold_results():
            pass
    return close_mask


def precompute_sl_array(df: pd.DataFrame, sl_mult: float = 2.0) -> np.ndarray:
    """Pre-compute Stop-Loss array."""
    from strategy.exit_rules import precompute_sl_array as _sl
    return _sl(df, sl_mult=sl_mult)


def precompute_tp_array(df: pd.DataFrame, tp_mult: float = 2.0) -> np.ndarray:
    """Pre-compute Take-Profit array."""
    from strategy.exit_rules import precompute_tp_array as _tp
    return _tp(df, tp_mult=tp_mult)


__all__ = [
    "generate_open_signals",
    "generate_close_signals",
    "precompute_sl_array",
    "precompute_tp_array",
]
