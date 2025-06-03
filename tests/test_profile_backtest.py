import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import profile_backtest
import logging


def test_main_profile_runs(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=3, freq='min', tz='UTC'),
        'Open': [1, 2, 3],
        'High': [1, 2, 3],
        'Low': [1, 2, 3],
        'Close': [1, 2, 3]
    })
    csv_path = tmp_path / 'mini.csv'
    df.to_csv(csv_path, index=False)

    captured_cols = {}

    def dummy_run_backtest(df, *args, **kwargs):
        captured_cols['cols'] = df.columns.tolist()

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)

    profile_backtest.main_profile(str(csv_path), num_rows=2)

    for required in ['ATR_14', 'Gain_Z', 'MACD_hist']:
        assert required in captured_cols['cols']


def test_main_profile_missing_columns(tmp_path, caplog):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=3, freq='min', tz='UTC'),
        'Open': [1, 2, 3],
        'Close': [1, 2, 3]
    })
    csv_path = tmp_path / 'missing.csv'
    df.to_csv(csv_path, index=False)
    caplog.set_level(logging.ERROR)
    profile_backtest.main_profile(str(csv_path), num_rows=2)
    assert "Missing required columns" in caplog.text
