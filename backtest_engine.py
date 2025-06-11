"""
Module: backtest_engine.py
Provides a function to regenerate the trade log via your core backtest simulation.
"""
import pandas as pd

from src.config import DATA_FILE_PATH_M1, DATA_FILE_PATH_M15
from src.strategy import run_backtest_simulation_v34
from src.features import engineer_m1_features, calculate_m15_trend_zone, calculate_m1_entry_signals

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
        # [Patch v6.5.17] specify format to silence pandas warnings
        df = pd.read_csv(
            DATA_FILE_PATH_M1,
            index_col=0,
            parse_dates=[0],
            date_format="%Y%m%d",
        )
    except Exception as e:
        raise RuntimeError(f"[backtest_engine] Failed to load price data: {e}") from e

    # 1b) Ensure index is a DatetimeIndex so `.tz` attribute exists
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # [Patch v6.5.17] enforce format when converting index
            df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        except Exception as e:
            raise RuntimeError(
                f"[backtest_engine] Failed to convert index to datetime: {e}"
            ) from e

    # [Patch v6.5.15] Engineer features before simulation
    features_df = engineer_m1_features(df)
    # [Patch v6.6.0] Generate Trend Zone and entry signal features
    try:
        df_m15 = pd.read_csv(
            DATA_FILE_PATH_M15,
            index_col=0,
            parse_dates=[0],
            infer_datetime_format=True,
        )
    except Exception:
        df_m15 = None
    if df_m15 is not None and not df_m15.empty:
        # Ensure M15 index is a DatetimeIndex
        if not isinstance(df_m15.index, pd.DatetimeIndex):
            df_m15.index = pd.to_datetime(df_m15.index, infer_datetime_format=True)
        try:
            trend_df = calculate_m15_trend_zone(df_m15)
        except Exception:
            # Fallback: default all zones to NEUTRAL on error
            trend_df = pd.DataFrame("NEUTRAL", index=features_df.index, columns=["Trend_Zone"])
        # Align Trend_Zone values to M1 timeline
        trend_series = trend_df["Trend_Zone"].reindex(features_df.index, method="ffill").fillna("NEUTRAL")
        features_df["Trend_Zone"] = pd.Categorical(trend_series, categories=["NEUTRAL", "UP", "DOWN"])
    else:
        # No M15 data: assume neutral trend for all
        features_df["Trend_Zone"] = pd.Categorical(["NEUTRAL"] * len(features_df), categories=["NEUTRAL", "UP", "DOWN"])
    # Compute entry signals and related columns (Entry_Long, Entry_Short, Trade_Tag, Signal_Score, Trade_Reason)
    from src.strategy import ENTRY_CONFIG_PER_FOLD
    base_config = ENTRY_CONFIG_PER_FOLD.get(0, {})
    features_df = calculate_m1_entry_signals(features_df, base_config)

    # 3) Run your core simulation (returns tuple: (sim_df, trade_log_df, …))
    result = run_backtest_simulation_v34(
        features_df,
        label="regen",
        initial_capital_segment=100.0,
        fold_config=DEFAULT_FOLD_CONFIG,
        current_fold_index=DEFAULT_FOLD_INDEX,
    )

    # 4) Extract and validate the trade log DataFrame
    try:
        trade_log_df = result[1]
    except Exception:
        raise RuntimeError("[backtest_engine] Unexpected return format from simulation.")

    if trade_log_df is None or trade_log_df.empty:
        raise RuntimeError("[backtest_engine] Simulation produced an empty trade log.")

    return trade_log_df
