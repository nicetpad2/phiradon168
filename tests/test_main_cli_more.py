import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline
from src.utils.pipeline_config import PipelineConfig


def test_parse_args_profile():
    args = pipeline.parse_args(['--mode', 'backtest', '--profile', '--output-file', 'out.prof'])
    assert args.mode == 'backtest'
    assert args.profile
    assert args.output_file == 'out.prof'


def test_run_backtest_no_models_threshold(tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path), threshold_file='missing.csv')
    captured = {}
    pipeline.run_backtest(cfg, pipeline_func=lambda *a: captured.update(model=a[2], thresh=a[3]))
    assert captured['model'] is None
    assert captured['thresh'] is None


def test_run_backtest_threshold_no_median(tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path), threshold_file='th.csv')
    (tmp_path / 'model_1.joblib').write_text('x')
    pd.DataFrame({'value': [1]}).to_csv(tmp_path / 'th.csv', index=False)
    captured = {}
    pipeline.run_backtest(cfg, pipeline_func=lambda *a: captured.update(model=a[2], thresh=a[3]))
    assert captured['model'].endswith('model_1.joblib')
    assert captured['thresh'] is None


def test_run_backtest_best_threshold(tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path), threshold_file='th.csv')
    (tmp_path / 'model_1.joblib').write_text('x')
    pd.DataFrame({'best_threshold': [0.42]}).to_csv(tmp_path / 'th.csv', index=False)
    captured = {}
    pipeline.run_backtest(cfg, pipeline_func=lambda *a: captured.update(model=a[2], thresh=a[3]))
    assert captured['model'].endswith('model_1.joblib')
    assert captured['thresh'] == pytest.approx(0.42)
