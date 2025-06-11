import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows by index or Date/Timestamp."""
    if df.index.has_duplicates:
        dupes = df.index.duplicated().sum()
        logger.info("[Patch] Removing %s duplicated index rows", dupes)
        df = df[~df.index.duplicated(keep="first")]

    subset = ["Date", "Timestamp"]
    if all(col in df.columns for col in subset):
        dupes = df.duplicated(subset=subset).sum()
        if dupes:
            logger.info("[Patch] Removing %s duplicated rows by Date/Timestamp", dupes)
            df = df.drop_duplicates(subset=subset, keep="first")
    else:
        dupes = df.duplicated().sum()
        if dupes:
            logger.info("[Patch] Removing %s duplicated rows", dupes)
            df = df.drop_duplicates()
    return df


def clean_csv(path: str, output: str | None = None) -> None:
    """Load CSV, remove duplicates and save to output."""
    df = pd.read_csv(path)
    cleaned = remove_duplicates(df)
    out_path = output or path
    cleaned.to_csv(out_path, index=False)
    logger.info("[Patch] Cleaned CSV written to %s", out_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CSV data cleaner")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("--output", help="Output path", default=None)
    args = parser.parse_args(argv)
    clean_csv(args.input, args.output)


if __name__ == "__main__":
    main()
