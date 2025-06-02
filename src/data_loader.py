"""Data loading utilities."""

from typing import Tuple
import pandas as pd

from .config import CSV_PATH_M1, CSV_PATH_M15


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load M1 and M15 CSV data.

    Returns:
        Tuple containing M1 dataframe and M15 dataframe.
    """
    df_m1 = pd.read_csv(CSV_PATH_M1)
    df_m15 = pd.read_csv(CSV_PATH_M15)
    return df_m1, df_m15

__all__ = ["load_data"]
