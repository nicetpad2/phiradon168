"""Profile the backtest simulation using cProfile.

This script loads a sample M1 dataset and runs ``run_backtest_simulation_v34``
while collecting profiling statistics. It helps identify performance
bottlenecks in the backtesting loop or feature calculations.
"""

import argparse
import cProfile
import pstats
import pandas as pd

from src.strategy import run_backtest_simulation_v34
from src.data_loader import safe_load_csv_auto


def main_profile(csv_path: str, num_rows: int = 5000) -> None:
    """Run the backtest simulation with profiling enabled."""
    df = safe_load_csv_auto(csv_path)
    if df is None:
        raise FileNotFoundError(f"File not found: {csv_path}")
    if not df.empty:
        df = df.head(num_rows)
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        else:
            # Fallback for files where the first column becomes the index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
    run_backtest_simulation_v34(df, label="profile", initial_capital_segment=10000)


def profile_from_cli() -> None:
    parser = argparse.ArgumentParser(description="Profile backtest simulation")
    parser.add_argument('csv', help='Path to M1 data CSV')
    parser.add_argument('--rows', type=int, default=5000, help='Number of rows to load')
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    main_profile(args.csv, args.rows)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)


if __name__ == '__main__':
    profile_from_cli()
