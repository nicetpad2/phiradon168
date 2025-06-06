"""Utility plotting functions for strategy visuals."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["plot_equity_curve"]


def plot_equity_curve(trade_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot an equity curve and save it to the given directory.

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
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(trade_df.index, trade_df.squeeze())
    ax.set_xlabel("Index")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    file_path = output_path / f"equity_curve_{timestamp}.png"
    fig.savefig(file_path, dpi=300)
    plt.close(fig)
    return file_path
