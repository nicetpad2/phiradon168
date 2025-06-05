import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def test_fake_trade_generated(monkeypatch, tmp_path):
    os.environ['DEBUG_FAKE_TRADE'] = '1'

    df = pd.DataFrame({
        'Open': [1]*6,
        'High': [1]*6,
        'Low': [1]*6,
        'Close': [1]*6,
        'ATR_14_Shifted': [0.1]*6,
    }, index=pd.date_range('2023-01-01', periods=6, freq='min'))

    def dummy_run(*args, **kwargs):
        return (df.iloc[:0], pd.DataFrame(), 1000.0, {}, 0.0, {}, [], 'L1', 'L2', False, 0, 0.0)

    monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)

    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    fund = {'name': 'DBG', 'mm_mode': 'static', 'risk': 1}

    result = strategy.run_all_folds_with_threshold(
        fund_profile=fund,
        df_m1_final=df,
        n_walk_forward_splits=2,
        output_dir=str(out_dir)
    )

    trade_log = result[3]
    assert trade_log.empty
    os.environ.pop('DEBUG_FAKE_TRADE')

