"""Plot helpers for strategy results."""
from __future__ import annotations

from typing import Sequence, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_equity_curve(equity: Sequence[float], filepath: Optional[str] = None):
    """Plot equity curve and optionally save to file."""
    fig, ax = plt.subplots()
    ax.plot(list(equity))
    ax.set_title("Equity Curve")
    if filepath:
        fig.savefig(filepath)
    return fig

__all__ = ["plot_equity_curve"]
