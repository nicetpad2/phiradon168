"""Utility plotting functions for strategy visuals."""
import matplotlib.pyplot as plt
from typing import Sequence

__all__ = ["plot_equity_curve"]


def plot_equity_curve(equity: Sequence[float]):
    """Plot a simple equity curve."""
    plt.figure()
    plt.plot(list(equity))
    plt.title("Equity Curve")
    plt.xlabel("Trade")
    plt.ylabel("Equity")
    return plt.gca()
