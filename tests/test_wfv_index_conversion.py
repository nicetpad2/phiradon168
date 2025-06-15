import os, sys
import pandas as pd
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def test_run_all_folds_index_conversion(monkeypatch, tmp_path, caplog):
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2023-01-01', periods=4, freq='min'),
        'Open': [1, 1, 1, 1],
        'High': [1, 1, 1, 1],
        'Low': [1, 1, 1, 1],
        'Close': [1, 1, 1, 1],
        'ATR_14_Shifted': [0.1]*4,
        'ATR_14': [0.1]*4,
        'ATR_14_Rolling_Avg': [0.1]*4,
    })
    # Use RangeIndex to trigger conversion
    df.reset_index(drop=True, inplace=True)

    def dummy_run(*args, **kwargs):
        trade_log = pd.DataFrame({'side': ['BUY'], 'exit_reason': ['TP']})
        return (df.iloc[:1].set_index(pd.to_datetime(df['Timestamp']).iloc[:1]), trade_log, 1000.0, {}, 0.0, {}, [], 'L1', 'L2', False, 0, 0.0)

    monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    fund = {'name': 'T', 'mm_mode': 'static', 'risk': 1}

    with caplog.at_level(logging.INFO):
        strategy.run_all_folds_with_threshold(
            fund_profile=fund,
            df_m1_final=df,
            n_walk_forward_splits=2,
            output_dir=str(out_dir)
        )
    assert "Successfully converted" in caplog.text
