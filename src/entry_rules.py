import os
import pandas as pd
from src import config
from src.constants import ColumnName, Signal

__all__ = ["generate_open_signals"]

# [Patch v6.5.9] Dynamic threshold + MA fallback

def generate_open_signals(df: pd.DataFrame, features: list[str] | None = None) -> pd.Series:
    """Generate binary open signals based on signal_score and MA crossover fallback."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if "signal_score" not in df.columns:
        raise KeyError("'signal_score' column required")
    threshold = float(os.getenv("MIN_SIGNAL_SCORE_ENTRY", config.MIN_SIGNAL_SCORE_ENTRY))
    df = df.copy()
    df["signal"] = (
        pd.to_numeric(df["signal_score"], errors="coerce") >= threshold
    ).astype(int) * Signal.LONG
    if df["signal"].sum() == 0 and ColumnName.CLOSE in df.columns:
        fast = df[ColumnName.CLOSE].rolling(config.FAST_MA_PERIOD, min_periods=1).mean()
        slow = df[ColumnName.CLOSE].rolling(config.SLOW_MA_PERIOD, min_periods=1).mean()
        df["signal"] = ((fast > slow) & (fast.shift() <= slow.shift())).astype(int) * Signal.LONG
    return df["signal"]
