"""[Patch v6.7.12] CSV validation and cleaning utilities."""

import argparse
import logging
from typing import Iterable

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


def validate_and_convert_csv(
    path: str,
    output: str | None = None,
    required_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """[Patch] Load, clean, validate then optionally save a CSV file."""
    if required_cols is None:
        required_cols = DEFAULT_REQUIRED_COLS
    df = data_cleaner.read_csv_auto(path)
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
