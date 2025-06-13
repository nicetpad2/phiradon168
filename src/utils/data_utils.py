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
    """Load CSV file gracefully even if ``timestamp`` column is missing."""
    if not os.path.exists(path):
        logger.error("prepare_csv_auto: file not found %s", path)
        logging.getLogger().error("prepare_csv_auto: file not found %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        logger.warning("[QA-WARNING] timestamp column missing in %s", path)
        logging.getLogger().warning("[QA-WARNING] timestamp column missing in %s", path)
    return df


def parse_thai_date_fast(date_series: pd.Series) -> pd.Series:
    """Convert Thai Buddhist date strings to Gregorian Timestamps."""
    parts = date_series.str.split(" ", n=1, expand=True)
    date_only = parts[0]
    time_part = pd.to_timedelta(parts[1], errors="coerce") if len(parts.columns) > 1 else None

    if date_only.str.contains("-").any():
        base = pd.to_datetime(date_only, errors="coerce")
    elif date_only.str.contains("/").any():
        date_parts = date_only.str.extract(r"(\d{1,2})/(\d{1,2})/(\d{4})", expand=True)
        date_parts.columns = ["day", "month", "year"]
        for col in date_parts.columns:
            date_parts[col] = pd.to_numeric(date_parts[col], errors="coerce")
        date_parts["year"] -= 543
        base = pd.to_datetime(date_parts[["year", "month", "day"]], errors="coerce")
    else:
        date_parts = date_only.str.extract(r"(\d{4})(\d{2})(\d{2})", expand=True)
        date_parts.columns = ["year", "month", "day"]
        for col in date_parts.columns:
            date_parts[col] = pd.to_numeric(date_parts[col], errors="coerce")
        date_parts["year"] -= 543
        base = pd.to_datetime(date_parts[["year", "month", "day"]], errors="coerce")

    if time_part is not None:
        base = base + time_part
    return base


__all__ = [
    "convert_thai_datetime",
    "prepare_csv_auto",
    "parse_thai_date_fast",
]
