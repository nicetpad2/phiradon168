import os
import sys
import pandas as pd
import logging
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

import src.main as main


def test_parse_arguments_returns_empty_dict():
    assert main.parse_arguments() == {}


def test_load_features_from_file_returns_empty_dict():
    assert main.load_features_from_file('missing.json') == {}


def test_run_pipeline_stage_preprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'OUTPUT_DIR', str(tmp_path))
    df = pd.DataFrame({'A': [1]})
    called = {}
    monkeypatch.setattr(main, 'load_data', lambda p, t, **kw: df)
    monkeypatch.setattr(main, 'engineer_m1_features', lambda d: d)
    def fake_save(df_in, path, fmt):
        Path(path).write_text('x')
        called['saved'] = fmt
    monkeypatch.setattr(main, 'save_features', fake_save)
    monkeypatch.setattr(main, 'maybe_collect', lambda: called.setdefault('collect', True))
    out = main.run_pipeline_stage('preprocess')
    assert Path(out).exists()
    assert called['saved'] == 'parquet' and called['collect']


def test_run_pipeline_stage_backtest_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'OUTPUT_DIR', str(tmp_path))
    preproc = Path(tmp_path) / 'preprocessed.parquet'
    preproc.write_text('dummy')
    df = pd.DataFrame({'A': [1]})
    monkeypatch.setattr(main, 'load_features', lambda p, f: df)
    called = {}
    monkeypatch.setitem(main.__dict__, 'run_backtest_simulation_v34', lambda d, label, initial_capital_segment: called.setdefault('run', True))
    main.run_pipeline_stage('backtest')
    assert called['run']


def test_run_pipeline_stage_report_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'OUTPUT_DIR', str(tmp_path))
    result = main.run_pipeline_stage('report')
    assert result is None


def test_run_pipeline_stage_report_with_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'OUTPUT_DIR', str(tmp_path))
    metrics = Path(tmp_path) / 'metrics_summary.csv'
    metrics.write_text('a,1\n')
    monkeypatch.setattr(pd, 'read_csv', lambda p: pd.DataFrame({'a': [1]}))
    called = {}
    monkeypatch.setattr(main, 'plot_equity_curve', lambda *a, **k: called.setdefault('plot', True))
    main.run_pipeline_stage('report')
    assert called['plot']


def test_run_pipeline_stage_unknown():
    assert main.run_pipeline_stage('unknown') is None
