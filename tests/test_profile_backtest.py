import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import profile_backtest


def test_main_profile_runs(tmp_path):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=3, freq='min', tz='UTC'),
        'Open': [1, 2, 3],
        'High': [1, 2, 3],
        'Low': [1, 2, 3],
        'Close': [1, 2, 3]
    })
    csv_path = tmp_path / 'mini.csv'
    df.to_csv(csv_path, index=False)
    profile_backtest.main_profile(str(csv_path), num_rows=2)
