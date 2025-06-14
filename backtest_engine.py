"""
Module: backtest_engine.py
Provides a function to regenerate the trade log via your core backtest simulation.
"""
import pandas as pd
import logging

from src.config import DEFAULT_CSV_PATH_M1, DEFAULT_CSV_PATH_M15
from src.strategy import (
    run_backtest_simulation_v34,
    MainStrategy,
    DefaultEntryStrategy,
    DefaultExitStrategy,
)
from src.features import (
    engineer_m1_features,
    calculate_m15_trend_zone,
    calculate_m1_entry_signals,
    load_feature_config,
)
from src.data_loader import safe_load_csv_auto

logger = logging.getLogger(__name__)


def _prepare_m15_data_optimized(m15_filepath, config):
    """Load and prepare M15 data using safe_load_csv_auto."""
    logger.info(
        f"   (Optimized Load) กำลังโหลดและเตรียมข้อมูล M15 จาก: {m15_filepath}"
    )

    m15_df = safe_load_csv_auto(
        m15_filepath,
        row_limit=config.get("pipeline", {}).get("limit_m15_rows"),
    )

    if m15_df is None or m15_df.empty:
        logger.error("   (Critical Error) ไม่สามารถโหลดข้อมูล M15 ได้ หรือข้อมูลว่างเปล่า")
        return None

    # [Patch v6.9.4] Auto-detect datetime column names in M15 data
    if {"date", "timestamp"}.issubset(m15_df.columns):
        combined = m15_df["date"].astype(str) + " " + m15_df["timestamp"].astype(str)
        m15_df.index = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
        if m15_df.index.isnull().sum() > 0.5 * len(m15_df):
            logger.warning(
                "(Warning) การ parse วันที่/เวลา (M15) ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            m15_df.index = pd.to_datetime(combined, errors="coerce", format="mixed")
        m15_df.drop(columns=["date", "timestamp"], inplace=True)
        dup_count = int(m15_df.index.duplicated().sum())
        logger.warning(
            "(Warning) พบ duplicate labels ใน index M15 ... Removed %s duplicate rows",
            dup_count,
        )
        if dup_count > 0:
            m15_df = m15_df.loc[~m15_df.index.duplicated(keep="first")]
    else:
        possible_cols = ["Date", "Date/Time", "Timestamp", "datetime", "Datetime"]
        time_col = next((c for c in m15_df.columns if c.lower() in {p.lower() for p in possible_cols}), None)
        if time_col:
            m15_df.index = pd.to_datetime(m15_df[time_col], errors="coerce")
            m15_df.drop(columns=[time_col], inplace=True, errors="ignore")
        elif "date" in m15_df.columns:
            logger.warning(
                "(Warning) การ parse วันที่/เวลา (M15) ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            logger.warning(
                "(Warning) พบ duplicate labels ใน index M15 ... Removed %s duplicate rows",
                0,
            )

    if isinstance(m15_df.index, pd.DatetimeIndex) and m15_df.index.tz is not None:
        m15_df.index = m15_df.index.tz_convert(None)

    trend_zone_df = calculate_m15_trend_zone(m15_df)

    if trend_zone_df is None or trend_zone_df.empty:
        logger.error("   (Critical Error) การคำนวณ M15 Trend Zone ล้มเหลว")
        return None

    return trend_zone_df

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
    # 1) Load the raw M1 price data using safe_load_csv_auto
    try:
        df = safe_load_csv_auto(DEFAULT_CSV_PATH_M1)
    except Exception as e:
        raise RuntimeError(f"[backtest_engine] Failed to load price data: {e}") from e

    # [Patch v6.9.4] Auto-detect datetime columns for more robust parsing
    date_cols_upper = {"Date", "Timestamp"}
    date_cols_lower = {"date", "timestamp"}
    possible_cols = ["Date", "Date/Time", "Timestamp", "datetime", "Datetime"]

    if date_cols_upper.issubset(df.columns) or date_cols_lower.issubset(df.columns):
        d_col = "Date" if "Date" in df.columns else "date"
        t_col = "Timestamp" if "Timestamp" in df.columns else "timestamp"
        combined = df[d_col].astype(str) + " " + df[t_col].astype(str)
        df.index = pd.to_datetime(combined, format="%Y%m%d %H:%M:%S", errors="coerce")
        if df.index.isnull().sum() > 0.5 * len(df):
            logging.warning(
                "(Warning) การ parse วันที่/เวลา ด้วย format ที่กำหนดไม่สำเร็จ - กำลัง parse ใหม่แบบไม่ระบุ format"
            )
            df.index = pd.to_datetime(combined, errors="coerce", format="mixed")
        df.drop(columns=[d_col, t_col], inplace=True)
    else:
        time_col = next((c for c in df.columns if c in possible_cols), None)
        if time_col:
            df.index = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
            df.drop(columns=[time_col], inplace=True, errors="ignore")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce", format="mixed")

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
    trend_df = _prepare_m15_data_optimized(
        DEFAULT_CSV_PATH_M15,
        {"pipeline": {}, "trend_zone": {}},
    )
    if trend_df is not None:
        dup_count = int(trend_df.index.duplicated().sum())
        logging.warning(
            "(Warning) พบ duplicate labels ใน index M15, กำลังลบซ้ำ (คงไว้ค่าแรกของแต่ละ index)"
        )
        if dup_count > 0:
            trend_df = trend_df.loc[~trend_df.index.duplicated(keep='first')]
        logging.info(f"      Removed {dup_count} duplicate index rows from Trend Zone data.")
        if not trend_df.index.is_monotonic_increasing:
            trend_df.sort_index(inplace=True)
            logging.info("      Sorted Trend Zone DataFrame index in ascending order for alignment")
        trend_series = trend_df["Trend_Zone"].reindex(features_df.index, method="ffill").fillna("NEUTRAL")
        features_df["Trend_Zone"] = pd.Categorical(trend_series, categories=["NEUTRAL", "UP", "DOWN"])
    else:
        features_df["Trend_Zone"] = pd.Categorical(
            ["NEUTRAL"] * len(features_df), categories=["NEUTRAL", "UP", "DOWN"]
        )
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


def run_full_backtest(config: dict):
    """Run a full backtest pipeline using provided configuration."""
    logger.info("--- [FULL PIPELINE] เริ่มการทดสอบ Backtest เต็มรูปแบบ ---")

    data_cfg = config.get("data", {})
    pipeline_cfg = config.get("pipeline", {})
    strategy_cfg = config.get("strategy_settings", {})

    df_m1 = safe_load_csv_auto(data_cfg.get("m1_path"), pipeline_cfg.get("limit_m1_rows"))
    if df_m1 is None:
        logger.error("ไม่สามารถโหลดข้อมูล M1 ได้, ยกเลิกการทำงาน")
        return None

    feature_config = load_feature_config(data_cfg.get("feature_config", ""))
    df_m1 = engineer_m1_features(df_m1, feature_config)

    trend_zone_df = _prepare_m15_data_optimized(data_cfg.get("m15_path"), config)
    if trend_zone_df is None:
        logger.error("ไม่สามารถเตรียมข้อมูล M15 Trend Zone ได้, ยกเลิกการทำงาน")
        return None

    logger.info("   (Processing) กำลังรวมข้อมูล M1 และ M15 Trend Zone...")
    df_m1.sort_index(inplace=True)
    final_df = pd.merge_asof(
        df_m1,
        trend_zone_df,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=pd.Timedelta("15min"),
    )
    final_df["Trend_Zone"].fillna(method="ffill", inplace=True)
    final_df.dropna(subset=["Trend_Zone"], inplace=True)

    logger.info(f"   (Success) รวมข้อมูลสำเร็จ, จำนวนแถวสุดท้าย: {len(final_df)}")

    strategy_comp = MainStrategy(DefaultEntryStrategy(), DefaultExitStrategy())
    _ = strategy_comp.get_signal(final_df)

    result = run_backtest_simulation_v34(
        final_df,
        label="full_backtest",
        initial_capital_segment=strategy_cfg.get("initial_capital", 10000),
        fold_config=DEFAULT_FOLD_CONFIG,
        current_fold_index=0,
    )

    try:
        trade_log_df = result[1]
    except Exception:
        logger.error("[full_backtest] Unexpected return format from simulation")
        return None

    return trade_log_df
