import argparse
import os
from pathlib import Path
from src.data_loader import auto_convert_csv_to_parquet

PROJECT_CSVS = ["XAUUSD_M1.csv", "XAUUSD_M15.csv"]


def convert_project_csvs(base_dir: str, dest_folder: str) -> list[str]:
    """Convert default project CSV files to Parquet.

    Parameters
    ----------
    base_dir : str
        Base directory containing the CSV files.
    dest_folder : str
        Destination folder to store converted Parquet files.

    Returns
    -------
    list[str]
        List of paths to created Parquet files. Empty if conversion skipped.
    """
    base = Path(base_dir)
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in PROJECT_CSVS:
        csv_path = base / name
        auto_convert_csv_to_parquet(str(csv_path), dest)
        parquet_path = dest / f"{csv_path.stem}.parquet"
        if parquet_path.exists():
            saved.append(str(parquet_path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert project CSVs to Parquet")
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(os.path.dirname(__file__)),
        help="Folder containing project CSV files",
    )
    parser.add_argument(
        "--dest",
        default="parquet",
        help="Destination folder for Parquet files",
    )
    args = parser.parse_args()
    files = convert_project_csvs(args.base_dir, args.dest)
    for f in files:
        print(f"Saved {f}")


if __name__ == "__main__":
    main()
