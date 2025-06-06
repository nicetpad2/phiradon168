import os
# [Patch] Additional tests for quick_qa_output
import sys
import pandas as pd
import gzip
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import qa_tools
import pytest

quick_qa_output = qa_tools.quick_qa_output


def test_quick_qa_output(tmp_path):
    f_ok = tmp_path / 'fold1.csv.gz'
    df_ok = pd.DataFrame({'pnl':[1], 'entry_price':[1]})
    with gzip.open(f_ok, 'wt') as fh:
        df_ok.to_csv(fh, index=False)

    f_missing = tmp_path / 'fold2.csv.gz'
    df_missing = pd.DataFrame({'pnl':[ ]})
    with gzip.open(f_missing, 'wt') as fh:
        df_missing.to_csv(fh, index=False)

    issues = quick_qa_output(str(tmp_path), 'report.txt')
    assert any('No trades' in i or 'Missing columns' in i for i in issues)
    report_path = Path(tmp_path) / 'report.txt'
    assert report_path.exists()


def test_quick_qa_output_missing_columns(tmp_path):
    f_bad = tmp_path / 'fold_missing.csv.gz'
    df_bad = pd.DataFrame({'pnl': [1, 2]})
    with gzip.open(f_bad, 'wt') as fh:
        df_bad.to_csv(fh, index=False)

    issues = quick_qa_output(str(tmp_path), 'missing_report.txt')
    assert any('Missing columns' in i for i in issues)


def test_quick_qa_output_error(tmp_path):
    corrupt_file = tmp_path / 'corrupt.csv.gz'
    corrupt_file.write_bytes(b'invalid')

    issues = quick_qa_output(str(tmp_path), 'error_report.txt')
    assert any('Error' in i for i in issues)


def test_run_noise_backtest_calls_backtest(monkeypatch):
    captured = {}

    def dummy_backtest(df, label, initial_capital_segment, **kwargs):
        captured['rows'] = len(df)
        trade_log = pd.DataFrame({'pnl_usd_net': [0.0]})
        return (df, trade_log, initial_capital_segment, {}, 0.0, {}, [], 'A', 'B', False, 0, 0.0)

    monkeypatch.setattr(qa_tools.strategy, 'run_backtest_simulation_v34', dummy_backtest)

    result = qa_tools.run_noise_backtest(n=50, seed=42)
    assert captured['rows'] == 50
    assert 'total_pnl' in result


def test_run_noise_backtest_metrics(monkeypatch):
    def dummy_backtest(df, label, initial_capital_segment, **kwargs):
        log = pd.DataFrame({'pnl_usd_net': [1.0, -0.5]})
        final_eq = initial_capital_segment + log['pnl_usd_net'].sum()
        return (df, log, final_eq, {}, 0.0, {}, [], 'A', 'B', False, 0, 0.0)

    monkeypatch.setattr(qa_tools.strategy, 'run_backtest_simulation_v34', dummy_backtest)

    res = qa_tools.run_noise_backtest(n=10)
    assert res['total_pnl'] == 0.5
    assert abs(res['winrate'] - 50.0) < 1e-6
