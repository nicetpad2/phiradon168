"""
Module: backtest_engine.py
Provides a function to regenerate the trade log via your core backtest simulation.
"""
import pandas as pd

from src.config import DATA_FILE_PATH_M1
from src.strategy import run_backtest_simulation_v34

# [Patch v6.5.14] Force fold 0 of 1 when regenerating the trade log
DEFAULT_FOLD_CONFIG = {"n_folds": 1}
DEFAULT_FOLD_INDEX = 0


def run_backtest_engine(features_df: pd.DataFrame) -> pd.DataFrame:
    """Regenerate the trade log when the existing CSV has too few rows.

    Args:
        features_df (pd.DataFrame): Loaded features (not used directly here,
                                     but signature maintained for compatibility).

    Returns:
        pd.DataFrame: A DataFrame of trades (timestamp, price, signal, etc.).
    """
    # 1) Load the raw M1 price data
    try:
        # [Patch] Use explicit date parsing for consistent datetime index
        df = pd.read_csv(
            DATA_FILE_PATH_M1,
            index_col=0,
            parse_dates=[0],
            infer_datetime_format=True,
        )
    except Exception as e:
        raise RuntimeError(f"[backtest_engine] Failed to load price data: {e}") from e

    # 1b) Ensure index is a DatetimeIndex so `.tz` attribute exists
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, infer_datetime_format=True)
        except Exception as e:
            raise RuntimeError(
                f"[backtest_engine] Failed to convert index to datetime: {e}"
            ) from e

    # 2) Run your core simulation (returns tuple: (sim_df, trade_log_df, …))
    result = run_backtest_simulation_v34(
        df,
        label="regen",
        initial_capital_segment=100.0,
        fold_config=DEFAULT_FOLD_CONFIG,
        current_fold_index=DEFAULT_FOLD_INDEX,
    )

    # 3) Extract and validate the trade log DataFrame
    try:
        trade_log_df = result[1]
    except Exception:
        raise RuntimeError("[backtest_engine] Unexpected return format from simulation.")

    if trade_log_df is None or trade_log_df.empty:
        raise RuntimeError("[backtest_engine] Simulation produced an empty trade log.")

    return trade_log_df
