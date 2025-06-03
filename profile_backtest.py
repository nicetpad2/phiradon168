"""Profile the backtest simulation using cProfile.

This script loads a sample M1 dataset and runs ``run_backtest_simulation_v34``
while collecting profiling statistics. It helps identify performance
bottlenecks in the backtesting loop or feature calculations.
"""

import argparse
import cProfile
import pstats
import pandas as pd
import logging

from src.strategy import run_backtest_simulation_v34
from src.data_loader import safe_load_csv_auto
from src.features import engineer_m1_features  # [Patch v5.1.5]

logger = logging.getLogger(__name__)


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

        # (1) Engineer all M1 features [Patch v5.1.5]
        df = engineer_m1_features(df)

        # (2) Merge M15 Trend Zone if available [Patch v5.1.6]
        m15_csv = csv_path.replace('_M1.csv', '_M15.csv')
        try:
            df15 = safe_load_csv_auto(m15_csv)
            if df15 is not None and not df15.empty:
                df15.index = pd.to_datetime(df15.index, errors='coerce')
                from src.features import calculate_m15_trend_zone
                df15 = calculate_m15_trend_zone(df15)
                df = pd.merge_asof(
                    df.reset_index().rename(columns={'index': 'Datetime'}),
                    df15.reset_index().rename(columns={'index': 'Datetime', 'Trend_Zone': 'M15_Trend_Zone'}),
                    on='Datetime',
                    direction='backward',
                    tolerance=pd.Timedelta(minutes=15)
                ).set_index('Datetime')
                df['Trend_Zone'] = df.pop('M15_Trend_Zone')
        except FileNotFoundError:
            logger.warning(f"(Warning) M15 file not found: {m15_csv} – skipping Trend_Zone merge.")

        # (3) Stub entry/exit columns so backtester won't crash [Patch v5.1.6]
        df['Entry_Long'] = False
        df['Entry_Short'] = False
        df['Signal_Score'] = 0.0
        df['Trade_Tag'] = ''
        df['Trade_Reason'] = ''
    # [Patch v5.1.0] ตรวจสอบคอลัมน์หลักก่อนเรียก backtest
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(
            f"(Error) Missing required columns in input DataFrame for profile: {missing}"
        )
        return
    # Provide minimal fold_config and current_fold_index to avoid warnings
    run_backtest_simulation_v34(
        df,
        label="profile",
        initial_capital_segment=10000,
        fold_config={},
        current_fold_index=0,
    )


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
