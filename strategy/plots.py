"""Utility plotting functions for strategy visuals."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    trade_df: pd.DataFrame | Sequence[float], output_path: Path | None = None
) -> Path | plt.Axes:
    """Plot an equity curve and optionally save it.

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

    if isinstance(trade_df, pd.DataFrame):
        equity = trade_df.get("Equity", trade_df.iloc[:, 0])
    else:
        equity = pd.Series(trade_df)

    fig, ax = plt.subplots()
    ax.plot(pd.Series(equity).tolist())
    ax.set_title("Equity Curve")

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_path / f"equity_{timestamp}.png"
        fig.savefig(filepath)
        return filepath
    return ax


__all__ = ["plot_equity_curve"]
