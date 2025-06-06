import pandas as pd
import numpy as np
from src.features import macd, rsi, detect_macd_divergence, sma
from src.config import USE_MACD_SIGNALS, USE_RSI_SIGNALS


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
    """สร้างสัญญาณเปิด order พร้อมตัวเลือกเปิด/ปิด MACD/RSI และตัวกรอง MTF"""

    price_cond = df["Close"] > df["Close"].shift(1)
    signals = [price_cond]

    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        macd_cond = df["MACD_hist"] > 0
        if detect_macd_divergence(df["Close"], df["MACD_hist"]) != "bull":
            macd_cond &= False
        signals.append(macd_cond)
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        rsi_cond = df["RSI"] > 50
        signals.append(rsi_cond)

    if "MA_fast" not in df.columns:
        df = df.copy()
        df["MA_fast"] = sma(df["Close"], ma_fast)
    if "MA_slow" not in df.columns:
        df = df.copy()
        df["MA_slow"] = sma(df["Close"], ma_slow)
    ma_cond = df["MA_fast"] > df["MA_slow"]
    signals.append(ma_cond)

    if volume_col in df.columns:
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        avg_vol = vol.rolling(vol_window, min_periods=1).mean()
        vol_cond = vol > avg_vol * 1.5
        signals.append(vol_cond)

    signal_strength = sum(sig.astype(int) for sig in signals)
    open_mask = price_cond & (signal_strength >= 2)

    if trend == "UP":
        open_mask &= True
    elif trend == "DOWN":
        open_mask &= False

    return open_mask.fillna(0).astype(np.int8).to_numpy()
