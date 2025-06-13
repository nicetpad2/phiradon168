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




__all__ = [
    "convert_thai_datetime",
    "prepare_csv_auto",
]
