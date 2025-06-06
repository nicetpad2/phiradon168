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
    csv_path = tmp_path / 'mini_M1.csv'
    df.to_csv(csv_path, index=False)

    captured_cols = {}

    def dummy_run_backtest(df, *args, **kwargs):
        captured_cols['cols'] = df.columns.tolist()

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)

    profile_backtest.main_profile(str(csv_path), num_rows=2)

    for required in ['ATR_14', 'Gain_Z', 'MACD_hist',
                     'Entry_Long', 'Entry_Short', 'Signal_Score',
                     'Trade_Tag', 'Trade_Reason']:
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


def test_main_profile_merges_m15(tmp_path, monkeypatch):
    idx_m1 = pd.date_range('2022-01-01', periods=4, freq='1min', tz='UTC')
    df_m1 = pd.DataFrame({
        'Datetime': idx_m1,
        'Open': [1, 2, 3, 4],
        'High': [1, 2, 3, 4],
        'Low': [1, 2, 3, 4],
        'Close': [1, 2, 3, 4]
    })
    m1_path = tmp_path / 'XAUUSD_M1.csv'
    df_m1.to_csv(m1_path, index=False)

    idx_m15 = pd.date_range('2022-01-01', periods=2, freq='15min', tz='UTC')
    df_m15 = pd.DataFrame({
        'Datetime': idx_m15,
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2]
    })
    m15_path = tmp_path / 'XAUUSD_M15.csv'
    df_m15.to_csv(m15_path, index=False)

    captured_df = {}

    def dummy_run_backtest(df, *args, **kwargs):
        captured_df['df'] = df

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)

    profile_backtest.main_profile(str(m1_path), num_rows=4)

    assert 'Trend_Zone' in captured_df['df'].columns


def test_main_profile_numeric_index(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [1, 2, 3],
        'Low': [1, 2, 3],
        'Close': [1, 2, 3]
    })
    csv_path = tmp_path / 'numeric_M1.csv'
    df.to_csv(csv_path)

    captured = {}

    def dummy_run_backtest(df, *args, **kwargs):
        captured['index'] = df.index

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run_backtest)

    profile_backtest.main_profile(str(csv_path), num_rows=3)

    assert isinstance(captured['index'], pd.DatetimeIndex)
    assert captured['index'].name == 'Datetime'


def test_profile_cli_output_file(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2]
    })
    m1 = tmp_path / 'mini_M1.csv'
    df.to_csv(m1, index=False)

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)

    out_txt = tmp_path / 'stats.txt'
    out_prof = tmp_path / 'run.prof'
    monkeypatch.setattr(sys, 'argv', [
        'profile_backtest.py', str(m1), '--rows', '2', '--limit', '5',
        '--output', str(out_txt), '--output-file', str(out_prof)
    ])

    profile_backtest.profile_from_cli()

    assert out_txt.is_file()
    assert out_prof.is_file()
    text = out_txt.read_text()
    assert 'ncalls' in text


def test_profile_cli_output_dir(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
        'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]
    })
    m1 = tmp_path / 'mini_M1.csv'
    df.to_csv(m1, index=False)

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)

    out_dir = tmp_path / 'profiles'
    monkeypatch.setattr(sys, 'argv', [
        'profile_backtest.py', str(m1), '--rows', '2', '--output-profile-dir', str(out_dir)
    ])

    profile_backtest.profile_from_cli()

    prof_files = list(out_dir.glob('*.prof'))
    assert len(prof_files) == 1


def test_main_profile_custom_fund(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2]
    })
    csv_path = tmp_path / 'fund_M1.csv'
    df.to_csv(csv_path, index=False)

    captured = {}

    def dummy_run(df, *args, **kwargs):
        captured['fund'] = kwargs.get('fund_profile')

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', dummy_run)

    profile_backtest.main_profile(str(csv_path), num_rows=2, fund_profile_name='AGGRESSIVE')

    assert captured['fund']['mm_mode'] == 'high_freq'


def test_main_profile_train_option(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2]
    })
    csv_path = tmp_path / 'train_M1.csv'
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)

    called = {}

    def dummy_train(out, *args, **kwargs):
        called['out'] = out
        return {}

    monkeypatch.setattr(profile_backtest, 'real_train_func', dummy_train)

    out_dir = tmp_path / 'models'
    profile_backtest.main_profile(str(csv_path), num_rows=2, train=True, train_output=str(out_dir))

    assert called['out'] == str(out_dir)


def test_profile_cli_fund_and_train(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
        'Open': [1, 2],
        'High': [1, 2],
        'Low': [1, 2],
        'Close': [1, 2]
    })
    m1 = tmp_path / 'mini_M1.csv'
    df.to_csv(m1, index=False)

    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)

    called = {}

    def dummy_train(out, *a, **k):
        called['out'] = out
        return {}

    monkeypatch.setattr(profile_backtest, 'real_train_func', dummy_train)

    out = tmp_path / 'stats.txt'
    prof = tmp_path / 'cli.prof'
    train_dir = tmp_path / 'models'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'profile_backtest.py', str(m1), '--rows', '2', '--limit', '5',
            '--output', str(out), '--output-file', str(prof), '--fund', 'SPIKE',
            '--train', '--train-output', str(train_dir)
        ],
    )

    profile_backtest.profile_from_cli()

    assert out.is_file()
    assert prof.is_file()
    assert called['out'] == str(train_dir)


def test_cli_console_level(monkeypatch, tmp_path):
    df = pd.DataFrame({'Datetime': pd.date_range('2022-01-01', periods=2, freq='min', tz='UTC'),
                       'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]})
    m1 = tmp_path / 'mini_M1.csv'
    df.to_csv(m1, index=False)
    monkeypatch.setattr(profile_backtest, 'run_backtest_simulation_v34', lambda *a, **k: None)
    monkeypatch.setattr(sys, 'argv', [
        'profile_backtest.py', str(m1), '--rows', '2', '--console_level', 'WARNING',
        '--output-file', str(tmp_path / 'console.prof')
    ])
    profile_backtest.profile_from_cli()
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            assert h.level == logging.WARNING

