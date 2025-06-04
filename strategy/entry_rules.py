import pandas as pd
import numpy as np
from src.features import macd, rsi, detect_macd_divergence
from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS


def generate_open_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """สร้างสัญญาณเปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
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
    return open_mask.fillna(0).astype(np.int8).to_numpy()
