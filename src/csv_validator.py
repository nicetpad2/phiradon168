"""[Patch v6.7.12] CSV validation and cleaning utilities."""

import argparse
import logging
from typing import Iterable

POSSIBLE_DATETIME_COLS = [
    "Time",
    "Date/Time",
    "Timestamp",
    "DateTime",
    "Datetime",
    "datetime",
    "date/time",
    "time",
]

DEFAULT_REQUIRED_COLS = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

import pandas as pd

from src import data_cleaner
from src.data_loader import validate_csv_data

logger = logging.getLogger(__name__)


def find_and_rename_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """[Patch v6.9.41] Find and standardize datetime column names."""
    datetime_col_found = None
    for col_name in POSSIBLE_DATETIME_COLS:
        if col_name in df.columns:
            datetime_col_found = col_name
            break

    if datetime_col_found:
        if datetime_col_found != "Time":
            df.rename(columns={datetime_col_found: "Time"}, inplace=True)
            logger.info("[Patch] Renamed column '%s' to 'Time'", datetime_col_found)
    elif {"Date", "Time"}.issubset(df.columns):
        df["Time"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
        logger.info("[Patch] Combined 'Date' and 'Time' into 'Time'")
    else:
        raise ValueError("Critical Error: Datetime column not found in CSV.")

    return df


def validate_and_convert_csv(
    path: str,
    output: str | None = None,
    required_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """[Patch] Load, clean, validate then optionally save a CSV file."""
    if required_cols is None:
        required_cols = DEFAULT_REQUIRED_COLS
    df = data_cleaner.read_csv_auto(path)
    # [Patch v6.9.41] ใช้ฟังก์ชันค้นหาและเปลี่ยนชื่อคอลัมน์เวลาให้เป็นมาตรฐาน
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df = find_and_rename_datetime_column(df)
    if "Timestamp" not in df.columns and "Time" in df.columns:
        df["Timestamp"] = df["Time"]

    if "Date" not in df.columns and "Timestamp" in df.columns:
        parts = df["Timestamp"].astype(str).str.split(" ", n=1, expand=True)
        if parts.shape[1] == 2:
            date_part = parts[0].str.replace("-", "")
            year = date_part.str[:4].astype(int, errors="ignore")
            date_part = year.where(year < 2500, year - 543).astype(str).str.zfill(4) + date_part.str[4:]
            df["Date"] = date_part
            df["Timestamp"] = parts[1]
            logger.info("[Patch] Split combined Timestamp into 'Date' and 'Timestamp'")
    # validate raw columns first to ensure expected structure
    validate_csv_data(df, required_cols)
    df = data_cleaner.clean_dataframe(df)
    if output:
        df.to_csv(output, index=False)
        logger.info("[Patch] Validated CSV written to %s", output)
    return df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Validate and clean CSV")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file", default=None)
    parser.add_argument("--require", nargs="*", default=None, help="Required columns")
    args = parser.parse_args(argv)
    validate_and_convert_csv(args.input, args.output, args.require)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
