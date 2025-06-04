import pandas as pd
import numpy as np
from src.features import macd, rsi
from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS


def generate_close_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """สร้างสัญญาณปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    close_mask = df["Close"] < df["Close"].shift(1)
    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        close_mask &= df["MACD_hist"] < 0
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        close_mask &= df["RSI"] < 50
    return close_mask.fillna(0).astype(np.int8).to_numpy()


def precompute_sl_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Stop-Loss ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)


def precompute_tp_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Take-Profit ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)
