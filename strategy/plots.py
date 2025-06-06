"""Utility plotting functions for strategy visuals."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(trade_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot an equity curve and save it with a timestamp.

    Parameters
    ----------
    trade_df : pd.DataFrame
        DataFrame ที่ประกอบด้วยคอลัมน์ตัวเลขของค่า equity
    output_path : Path
        โฟลเดอร์ปลายทางสำหรับบันทึกไฟล์รูป

    Returns
    -------
    Path
        พาธไฟล์ PNG ที่สร้างขึ้น
    """

    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots()

    ax.plot(list(equity))
    ax.set_title("Equity Curve")
    if filepath:
        fig.savefig(filepath)
    return ax


__all__ = ["plot_equity_curve"]
