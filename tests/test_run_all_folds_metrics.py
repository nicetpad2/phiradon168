import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import pandas as pd
from src import strategy


def test_run_all_folds_metrics_error(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'Open': [1, 1, 1, 1],
        'High': [1, 1, 1, 1],
        'Low': [1, 1, 1, 1],
        'Close': [1, 1, 1, 1],
        'ATR_14_Shifted': [0.1, 0.1, 0.1, 0.1],
    }, index=pd.date_range('2023-01-01', periods=4, freq='min'))

    def dummy_run(*args, **kwargs):
        trade_log = pd.DataFrame({'side': ['BUY'], 'exit_reason': ['TP']})
        return (df.iloc[:1], trade_log, 1000.0, {}, 0.0, {}, [], 'L1', 'L2', False, 0, 0.0)

    call = {'n': 0}
    def dummy_metrics(*args, **kwargs):
        call['n'] += 1
        if call['n'] == 1:
            raise RuntimeError('fail')
        return {}

    monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)
    monkeypatch.setattr(strategy, 'calculate_metrics', dummy_metrics)

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    fund = {'name': 'TFund', 'mm_mode': 'static', 'risk': 1}

    result = strategy.run_all_folds_with_threshold(
        fund_profile=fund,
        df_m1_final=df,
        n_walk_forward_splits=2,
        output_dir=str(out_dir)
    )
    metrics_buy, metrics_sell = result[0], result[1]
    assert isinstance(metrics_buy, dict)
    assert isinstance(metrics_sell, dict)
