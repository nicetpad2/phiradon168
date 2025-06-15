# Helper functions extracted from src.main.py
import pandas as pd
from src.data_loader import setup_output_directory as dl_setup_output_directory


def parse_arguments():
    """Stubbed argument parser."""
    return {}


def setup_output_directory(base_dir, dir_name):
    """Stubbed setup_output_directory for main."""
    return dl_setup_output_directory(base_dir, dir_name)


def load_features_from_file(_):
    """Stubbed loader for saved features."""
    return {}


def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Stubbed NaN dropper."""
    return df.dropna()


def convert_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Stubbed float32 converter."""
    return df.astype("float32", errors="ignore")


def run_initial_backtest():
    """Stubbed initial backtest runner."""
    return None


def save_final_data(df: pd.DataFrame, path: str) -> None:
    """Stubbed data saver."""
    df.to_csv(path)
