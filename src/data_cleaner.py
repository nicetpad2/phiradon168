"""Simple CLI และฟังก์ชันสำหรับทำความสะอาดข้อมูลราคา."""

import argparse
import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_auto(path: str) -> pd.DataFrame:
    """[Patch] โหลด CSV โดยตรวจสอบตัวคั่นอัตโนมัติ"""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = "," if "," in first_line else r"\s+"
    return pd.read_csv(path, sep=delimiter, engine="python")


def convert_buddhist_year(
    df: pd.DataFrame,
    date_col: str = "Date",
    time_col: str = "Timestamp",
    out_col: str = "Time",
) -> pd.DataFrame:
    """[Patch] แปลงปี พ.ศ. เป็น ค.ศ. และรวมเป็นคอลัมน์ ``Time``"""
    if date_col not in df.columns or time_col not in df.columns:
        return df

    year = df[date_col].astype(str).str[:4].astype(int)
    year = year.where(year < 2500, year - 543)
    rest = df[date_col].astype(str).str[4:]
    dt_str = (
        year.astype(str).str.zfill(4)
        + rest
        + " "
        + df[time_col].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df[out_col] = pd.to_datetime(dt_str, format="%Y%m%d %H:%M:%S", errors="coerce")
    df.drop(columns=[date_col, time_col], inplace=True)
    return df


def remove_duplicate_times(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """[Patch] ลบแถวที่มีเวลาซ้ำกัน"""
    if time_col in df.columns:
        dupes = df.duplicated(subset=time_col).sum()
        if dupes:
            logger.info("[Patch] Removing %s duplicated rows by %s", dupes, time_col)
            df = df.drop_duplicates(subset=time_col, keep="first")
    return df


def sort_by_time(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """[Patch] เรียงข้อมูลตามเวลา"""
    if time_col in df.columns:
        df = df.sort_values(time_col)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    method: str = "drop",
) -> pd.DataFrame:
    """[Patch] จัดการค่า NaN ในคอลัมน์ราคาหลัก"""
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]

    if method == "drop":
        df = df.dropna(subset=list(cols))
    else:
        means = df[list(cols)].mean()
        df[list(cols)] = df[list(cols)].fillna(means)
    return df


def validate_price_columns(df: pd.DataFrame, cols: Iterable[str] | None = None) -> None:
    """[Patch] ตรวจสอบว่าคอลัมน์ราคาครบถ้วนและเป็นตัวเลข"""
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error("Missing columns during validation: %s", missing)
        raise ValueError(f"Missing columns: {missing}")

    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            logger.error("Non-numeric column detected: %s", c)
            raise TypeError(f"Column {c} must be numeric")


def clean_dataframe(df: pd.DataFrame, fill_method: str = "drop") -> pd.DataFrame:
    """[Patch] ขั้นตอนทำความสะอาดข้อมูลแบบครบถ้วน"""
    logger.info(f"Rows before clean_dataframe: {len(df)}")
    df = convert_buddhist_year(df)
    logger.info(f"Rows after convert_buddhist_year: {len(df)}")
    df = remove_duplicate_times(df)
    logger.info(f"Rows after remove_duplicate_times: {len(df)}")
    df = sort_by_time(df)
    df = handle_missing_values(df, method=fill_method)
    logger.info(f"Rows after handle_missing_values: {len(df)}")
    try:
        validate_price_columns(df)
        logger.info("validate_price_columns passed")
    except Exception:
        logger.error("validate_price_columns failed", exc_info=True)
        raise
    logger.info("NaN count after clean_dataframe:\n%s", df.isna().sum().to_string())
    return df


def clean_csv(path: str, output: str | None = None, fill_method: str = "drop") -> None:
    """โหลด CSV แล้วทำความสะอาดข้อมูลก่อนบันทึก"""
    df = read_csv_auto(path)
    cleaned = clean_dataframe(df, fill_method=fill_method)
    out_path = output or path
    cleaned.to_csv(out_path, index=False)
    logger.info("[Patch] Cleaned CSV written to %s", out_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CSV data cleaner")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("--output", help="Output path", default=None)
    parser.add_argument(
        "--fill", choices=["drop", "mean"], default="drop", help="วิธีจัดการค่า NaN"
    )
    args = parser.parse_args(argv)
    clean_csv(args.input, args.output, args.fill)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

