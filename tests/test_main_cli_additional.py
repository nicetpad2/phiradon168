import os
import sys
import logging
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline


def test_run_backtest_pipeline_success(monkeypatch):
    called = {}
    import src.main as src_main
    monkeypatch.setattr(src_main, 'run_pipeline_stage', lambda s: called.setdefault('stage', s))
    pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), 'm.joblib', 0.1)
    assert called['stage'] == 'backtest'


def test_run_backtest_pipeline_exception(monkeypatch):
    logs = []
    monkeypatch.setattr(pipeline.logger, 'exception', lambda msg, *a, **k: logs.append(msg))
    import src.main as src_main

    def boom(stage):
        raise RuntimeError('fail')

    monkeypatch.setattr(src_main, 'run_pipeline_stage', boom)
    with pytest.raises(RuntimeError):
        pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), 'm.joblib', 0.1)
    assert any('Internal backtest error' in m for m in logs)


def test_main_report_gpu(monkeypatch):
    msgs = []
    monkeypatch.setattr(pipeline.logger, 'info', lambda msg, *a, **k: msgs.append(msg % a))
    monkeypatch.setattr(pipeline, 'run_report', lambda c: None)
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    monkeypatch.setattr(pipeline, 'has_gpu', lambda: True)
    res = pipeline.main(['--mode', 'report'])
    assert res == 0
    assert any('GPU detected' in m for m in msgs)
