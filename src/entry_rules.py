import os
import pandas as pd
from src import config

# [Patch v6.5.9] Dynamic threshold override via environment and MA crossover fallback


def generate_open_signals(
    df: pd.DataFrame, features: list[str] | None = None
) -> pd.Series:
    """Return entry signal series based on signal_score with optional MA fallback."""
    if df is None or "signal_score" not in df.columns:
        raise KeyError("DataFrame must contain 'signal_score'")
    df = df.copy()
    threshold = float(
        os.getenv("MIN_SIGNAL_SCORE_ENTRY", config.MIN_SIGNAL_SCORE_ENTRY)
    )
    df["signal"] = (
        pd.to_numeric(df["signal_score"], errors="coerce") >= threshold
    ).astype(int)
    if df["signal"].sum() == 0 and "close" in df.columns:
        df["fast_ma"] = df["close"].rolling(config.FAST_MA_PERIOD).mean()
        df["slow_ma"] = df["close"].rolling(config.SLOW_MA_PERIOD).mean()
        df["signal"] = (
            (df["fast_ma"] > df["slow_ma"])
            & (df["fast_ma"].shift() <= df["slow_ma"].shift())
        ).astype(int)
    return df["signal"]
