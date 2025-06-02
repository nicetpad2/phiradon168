"""Feature engineering utilities."""

import pandas as pd


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple moving average feature to dataframe."""
    if "Close" not in df.columns:
        raise KeyError("Column 'Close' not found in dataframe")
    df = df.copy()
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    return df

__all__ = ["add_simple_features"]
