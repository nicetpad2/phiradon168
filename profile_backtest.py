"""Profile the backtest simulation using cProfile.

This script loads a sample M1 dataset and runs ``run_backtest_simulation_v34``
while collecting profiling statistics. It helps identify performance
bottlenecks in the backtesting loop or feature calculations.
"""

import argparse
import cProfile
import pstats
import sys
import os
import time
import datetime
import importlib
import tracemalloc
import pandas as pd
import logging
from multiprocessing import get_context

from src.strategy import run_backtest_simulation_v34
from src.data_loader import safe_load_csv_auto
from src.features import engineer_m1_features, calculate_m1_entry_signals  # [Patch v5.1.5]

logger = logging.getLogger(__name__)


def run_profile(func, output_file: str) -> cProfile.Profile:
    """Run ``func`` under ``cProfile`` and dump results to ``output_file``."""
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        try:
            func()
        finally:
            profiler.disable()
    except ValueError:
        # Another profiler may already be active (e.g. pytest-cov)
        # Simply run the function without additional profiling
        func()
    profiler.dump_stats(output_file)
    return profiler


def calculate_features_for_fold(params):
    """Helper for multiprocessing Pool to calculate features."""
    symbol, df = params
    logger = logging.getLogger(__name__)
    logger.info("Calculating features for %s", symbol)
    return engineer_m1_features(df)


def run_parallel_feature_engineering(list_of_fold_params, processes=4):
    """Run feature engineering in parallel using multiprocessing Pool."""
    ctx = get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        results = pool.map(calculate_features_for_fold, list_of_fold_params)
    return results


def get_fund_profile(name: str | None) -> dict:
    """Return fund profile dict from config by name."""
    # [Patch v5.10.2] Import config lazily for pytest compatibility
    try:
        cfg = importlib.import_module("src.config")
        FUND_PROFILES = getattr(cfg, "FUND_PROFILES", {})
        DEFAULT_FUND_NAME = getattr(cfg, "DEFAULT_FUND_NAME", "NORMAL")
    except Exception:
        FUND_PROFILES = {"NORMAL": {"risk": 0.01, "mm_mode": "balanced"}}
        DEFAULT_FUND_NAME = "NORMAL"
    if not name:
        name = DEFAULT_FUND_NAME
    profile = FUND_PROFILES.get(name, FUND_PROFILES.get(DEFAULT_FUND_NAME, {}))
    profile = profile.copy()
    profile["name"] = name
    return profile


def main_profile(
    csv_path: str,
    num_rows: int = 5000,
    fund_profile_name: str | None = None,
    train: bool = False,
    train_output: str = "models",
) -> None:
    """Run the backtest simulation with profiling enabled."""
    if "_M1" not in os.path.basename(csv_path):
        guess = csv_path.replace("_M15.csv", "_M1.csv")
        if guess != csv_path and os.path.exists(guess):
            logger.warning(
                "(Warning) Provided CSV not M1; using '%s' instead", guess
            )
            csv_path = guess
        else:
            logger.warning(
                "(Warning) CSV path may not be M1 timeframe: %s", csv_path
            )
    df = safe_load_csv_auto(csv_path, row_limit=num_rows)
    if df is None:
        raise FileNotFoundError(f"File not found: {csv_path}")
    if not df.empty:
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        else:
            # Fallback for files where the first column becomes the index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            df.index.name = 'Datetime'

        # (1) Engineer all M1 features [Patch v5.1.5]
        df = engineer_m1_features(df)

        # (2) Merge M15 Trend Zone if available [Patch v5.1.6]
        m15_csv = csv_path.replace('_M1.csv', '_M15.csv')
        try:
            df15 = safe_load_csv_auto(m15_csv)
            if df15 is not None and not df15.empty:
                df15.index = pd.to_datetime(df15.index, errors='coerce')
                df15.index.name = 'Datetime'
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

        # (3) Calculate entry signals using default config
        from src.strategy import ENTRY_CONFIG_PER_FOLD
        base_cfg = ENTRY_CONFIG_PER_FOLD.get(0, {})
        df = calculate_m1_entry_signals(df, base_cfg)
    # [Patch v5.1.0] ตรวจสอบคอลัมน์หลักก่อนเรียก backtest
    required_cols = ['open', 'high', 'low', 'close']
    cols_lower = [c.lower() for c in df.columns]
    missing = [c for c in required_cols if c not in cols_lower]
    if missing:
        msg = f"(Error) Missing required columns in input DataFrame for profile: {missing}"
        logger.error(msg)
        logging.getLogger().error(msg)
        return
    # Provide minimal fold_config and current_fold_index to avoid warnings
    fund_profile = get_fund_profile(fund_profile_name)
    run_backtest_simulation_v34(
        df,
        label="profile",
        initial_capital_segment=10000,
        fold_config={},
        current_fold_index=0,
        fund_profile=fund_profile,
    )

    if train:
        # [Patch v5.10.2] Delay import to avoid side effects when testing
        from src.training import real_train_func
        real_train_func(train_output)


def profile_from_cli() -> None:
    parser = argparse.ArgumentParser(description="Profile backtest simulation")
    parser.add_argument('csv', help='Path to M1 data CSV')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to load')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (use fewer rows)')
    parser.add_argument('--limit', type=int, default=20, help='Number of functions to display')
    parser.add_argument('--output', help='File path to save the stats table')
    parser.add_argument('--output-file', default='backtest_profile.prof', help='Profiling result .prof file')
    parser.add_argument('--output-profile-dir', help='Directory to store profile files for each run')
    parser.add_argument('--fund', help='Fund profile name to use')
    parser.add_argument('--train', action='store_true', help='Run training after profiling')
    parser.add_argument('--train-output', default='models', help='Training output directory')
    parser.add_argument('--console_level', default='INFO', help='Console log level')
    args = parser.parse_args()
    num_rows = args.rows if args.rows is not None else (1000 if args.debug else None)
    level = getattr(logging, args.console_level.upper(), logging.INFO)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)

    # Determine output profile file path
    output_prof = args.output_file
    if args.output_profile_dir:
        os.makedirs(args.output_profile_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prof = os.path.join(args.output_profile_dir, f"{stamp}_{args.output_file}")

    tracemalloc.start()
    start = time.perf_counter()
    profiler = run_profile(
        lambda: main_profile(
            args.csv,
            num_rows,
            fund_profile_name=args.fund,
            train=args.train,
            train_output=args.train_output,
        ),
        output_prof,
    )
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    if args.output:
        with open(args.output, 'w') as f:
            stats.stream = f
            stats.print_stats(args.limit)
    else:
        stats.print_stats(args.limit)

    # Summary
    hottest = sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[:args.limit]
    hot_list = ", ".join(f"{h[0][2]}({h[1][3]:.2f}s)" for h in hottest)
    print(f"[SUMMARY] Runtime: {runtime:.2f}s, Peak memory: {peak/1024**2:.2f} MB")
    print(f"[HOT SPOTS] {hot_list}")


if __name__ == '__main__':
    profile_from_cli()
