"""
Module: backtest_engine.py
Provides a function to regenerate the trade log via your core backtest simulation.
"""
import pandas as pd
import logging

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
        df = pd.read_csv(DATA_FILE_PATH_M1)
    except Exception as e:
        raise RuntimeError(f"[backtest_engine] Failed to load price data: {e}") from e

    # [Patch v6.7.5] combine Date and Timestamp to unique datetime index if present
    if {"Date", "Timestamp"}.issubset(df.columns):
        combined = df["Date"].astype(str) + " " + df["Timestamp"].astype(str)
        df.index = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
        # [Patch v6.7.8] Fallback parse without explicit format when too many NaT
        if df.index.isnull().sum() > 0.5 * len(df):
            logging.warning(
                "(Warning) การ parse วันที่/เวลา ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            df.index = pd.to_datetime(combined, errors="coerce")
        df.drop(columns=["Date", "Timestamp"], inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # fallback: convert existing index
        df.index = pd.to_datetime(df.index, errors="coerce")

    # 1b) Ensure index is a DatetimeIndex so `.tz` attribute exists
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # [Patch v6.5.17] enforce format when converting index
            df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        except Exception as e:
            raise RuntimeError(
                f"[backtest_engine] Failed to convert index to datetime: {e}"
            ) from e

    # [Patch v6.6.5] Ensure M1 price index sorted and unique
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
        logging.warning(
            "(Warning) พบ index M1 ไม่เรียงลำดับเวลา กำลังจัดเรียงใหม่ในลำดับ ascending"
        )
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        logging.warning(
            "(Warning) พบ index ซ้ำซ้อนในข้อมูลราคา M1 กำลังลบรายการซ้ำ (คงไว้ค่าแรก)"
        )
        df = df.loc[~df.index.duplicated(keep='first')]
        logging.info(f"      Removed {dup_count} duplicate index rows from M1 data.")

    # [Patch v6.5.15] Engineer features before simulation
    features_df = engineer_m1_features(df)
    # [Patch v6.6.0] Generate Trend Zone and entry signal features
    try:
        df_m15 = pd.read_csv(DATA_FILE_PATH_M15)
    except Exception:
        df_m15 = None
    if df_m15 is not None and {"Date", "Timestamp"}.issubset(df_m15.columns):
        combined = df_m15["Date"].astype(str) + " " + df_m15["Timestamp"].astype(str)
        df_m15.index = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
        # [Patch v6.7.8] Fallback parse for M15 when explicit format fails
        if df_m15.index.isnull().sum() > 0.5 * len(df_m15):
            logging.warning(
                "(Warning) การ parse วันที่/เวลา (M15) ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            df_m15.index = pd.to_datetime(combined, errors="coerce")
        df_m15.drop(columns=["Date", "Timestamp"], inplace=True)
    elif df_m15 is not None and not isinstance(df_m15.index, pd.DatetimeIndex):
        df_m15.index = pd.to_datetime(df_m15.index, errors="coerce")
    if df_m15 is not None and not df_m15.empty:
        # Ensure M15 index is a DatetimeIndex
        if not isinstance(df_m15.index, pd.DatetimeIndex):
            # [Patch v6.6.1] enforce format when converting index
            df_m15.index = pd.to_datetime(df_m15.index, format="%Y%m%d", errors="coerce")
        try:
            trend_df = calculate_m15_trend_zone(df_m15)
        except Exception:
            # Fallback: assume NEUTRAL trend for all on error
            trend_df = pd.DataFrame("NEUTRAL", index=features_df.index, columns=["Trend_Zone"])
        # [Patch v6.6.3] Remove duplicate trend indices and sort index before alignment
        if trend_df.index.duplicated().any():
            dup_count = int(trend_df.index.duplicated().sum())
            logging.warning(
                "(Warning) พบ index ซ้ำซ้อนใน Trend Zone DataFrame, กำลังลบรายการซ้ำ (คงไว้ค่าแรกของแต่ละ index)"
            )
            trend_df = trend_df.loc[~trend_df.index.duplicated(keep='first')]
            logging.info(f"      Removed {dup_count} duplicate index rows from Trend Zone data.")
        if not trend_df.index.is_monotonic_increasing:
            trend_df.sort_index(inplace=True)
            logging.info("      Sorted Trend Zone DataFrame index in ascending order for alignment")
        # Align Trend_Zone values to M1 timeline by forward-filling last known trend; default missing to NEUTRAL
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
        # [Patch v6.7.6] Downgrade empty trade log to warning and return empty DataFrame
        logging.getLogger(__name__).warning(
            "[backtest_engine] Simulation produced an empty trade log. This might be expected if no entry signals were found."
        )
        return trade_log_df if trade_log_df is not None else pd.DataFrame()

    return trade_log_df
