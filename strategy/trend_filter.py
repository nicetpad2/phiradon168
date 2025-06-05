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
