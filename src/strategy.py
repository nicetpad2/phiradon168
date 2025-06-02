"""Basic strategy implementation."""

from typing import Tuple
import pandas as pd

from .features import add_simple_features


def simple_strategy(df: pd.DataFrame) -> Tuple[int, int]:
    """Run a naive strategy based on moving average crossover.

    Returns:
        Tuple of (number_of_buys, number_of_sells)
    """
    df = add_simple_features(df)
    buys = ((df["Close"] > df["sma_5"]).shift(1) & (df["Close"] <= df["sma_5"]))
    sells = ((df["Close"] < df["sma_5"]).shift(1) & (df["Close"] >= df["sma_5"]))
    num_buys = buys.sum()
    num_sells = sells.sum()
    return int(num_buys), int(num_sells)

__all__ = ["simple_strategy"]
