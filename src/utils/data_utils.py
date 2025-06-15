import logging
import os
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def convert_thai_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert Thai Buddhist year date strings in ``column`` to Gregorian."""
    if column not in df.columns:
        return df
    try:
        series = pd.to_datetime(df[column], errors="raise")
    except Exception as e:
        logger.error("convert_thai_datetime failed: %s", e, exc_info=True)
        series = pd.to_datetime(df[column], errors="coerce", format="mixed")
    df[column] = series
    return df


def prepare_csv_auto(path: str) -> pd.DataFrame:
    """Load a CSV file with Thai datetime conversion and basic cleaning."""
    if not os.path.exists(path):
        logger.error("prepare_csv_auto: file not found %s", path)
        logging.getLogger().error("prepare_csv_auto: file not found %s", path)
        return pd.DataFrame()

    # ใช้ตัวอ่าน CSV ที่ตรวจสอบตัวคั่นอัตโนมัติจาก data_cleaner
    from src.data_cleaner import read_csv_auto, convert_buddhist_year

    df = read_csv_auto(path)

    # Normalize common timestamp column names
    if "Timestamp" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"Timestamp": "timestamp"}, inplace=True)

    # กรณีมีคอลัมน์ Date/Timestamp ให้แปลงปี พ.ศ. และรวมเป็น timestamp
    if {"Date", "timestamp"}.issubset(df.columns):
        df.rename(columns={"timestamp": "Timestamp"}, inplace=True)
        df = convert_buddhist_year(df)
        df.rename(columns={"Time": "timestamp"}, inplace=True)

    if "timestamp" in df.columns:
        df = convert_thai_datetime(df, "timestamp")
        df.dropna(subset=["timestamp"], inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        dupes = df.duplicated(subset="timestamp").sum()
        if dupes:
            logger.info("[Patch] Removing %s duplicate timestamps", dupes)
            df = df.drop_duplicates(subset="timestamp", keep="last")
        df.set_index("timestamp", inplace=True)
    else:
        logger.warning("[QA-WARNING] timestamp column missing in %s", path)
        logging.getLogger().warning("[QA-WARNING] timestamp column missing in %s", path)

    return df


def safe_read_csv(path: str) -> pd.DataFrame:
    """Return DataFrame from ``path`` or empty DataFrame on error.

    Supports CSV and Parquet files automatically.
    """
    try:
        from src.data_cleaner import read_csv_auto, convert_buddhist_year

        if str(path).endswith(".parquet"):
            df = pd.read_parquet(path)
        elif str(path).endswith(".gz") or str(path).endswith(".zip"):
            df = pd.read_csv(path, compression="infer")
        else:
            df = read_csv_auto(path)

        if {"Date", "Timestamp"}.issubset(df.columns):
            df = convert_buddhist_year(df)
        return df
    except Exception as exc:  # pragma: no cover - unexpected IO error
        logger.error("safe_read_csv failed: %s", exc, exc_info=True)
        return pd.DataFrame()


__all__ = [
    "convert_thai_datetime",
    "prepare_csv_auto",
    "safe_read_csv",
]
