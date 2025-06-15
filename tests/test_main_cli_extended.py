import os
import sys
import subprocess
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline
from src.utils.pipeline_config import PipelineConfig


class DummyManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.order = []

    def run_all(self):  # pragma: no cover - simple order tracker
        self.order.extend(['load', 'sweep', 'wfv', 'save', 'qa'])


def test_parse_args_defaults():
    args = pipeline.parse_args([])
    assert args.mode == 'all'
    assert args.output_file == 'backtest_profile.prof'
    assert args.config.endswith('pipeline.yaml')


def test_parse_args_custom():
    args = pipeline.parse_args([
        '--mode', 'backtest',
        '--config', 'cfg.yaml',
        '--log-level', 'debug'
    ])
    assert args.mode == 'backtest'
    assert args.config == 'cfg.yaml'
    assert args.log_level == 'debug'


def test_parse_args_debug_only():
    args = pipeline.parse_args(['--debug'])
    assert args.debug
    assert args.rows is None


def test_parse_args_debug_rows():
    args = pipeline.parse_args(['--debug', '--rows', '123'])
    assert args.debug
    assert args.rows == 123


def test_parse_args_live_loop_default():
    args = pipeline.parse_args([])
    assert args.live_loop == 0


def test_run_preprocess_success():
    called = []
    def fake_run(cmd, check):
        called.append(cmd)
    pipeline.run_preprocess(PipelineConfig(), runner=fake_run)
    assert any('ProjectP.py' in c[1] for c in called)
    assert called[0][-2:] == ['--fill', 'drop']


def test_run_preprocess_calls_validator(monkeypatch):
    captured = {}
    monkeypatch.setattr(pipeline.csv_validator, 'validate_and_convert_csv', lambda p: captured.setdefault('path', p))
    pipeline.run_preprocess(PipelineConfig(), runner=lambda c, check: None)
    assert captured['path'].endswith('XAUUSD_M1.csv')


def test_run_preprocess_failure():
    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_preprocess(PipelineConfig(), runner=fake_run)


def test_run_preprocess_custom_method():
    called = []
    cfg = PipelineConfig(cleaning_fill_method='mean')
    def fake_run(cmd, check):
        called.append(cmd)
    pipeline.run_preprocess(cfg, runner=fake_run)
    assert called[0][-2:] == ['--fill', 'mean']


def test_run_sweep_success():
    marker = {}
    pipeline.run_sweep(PipelineConfig(), runner=lambda c, check: marker.setdefault('ok', True))
    assert marker['ok']


def test_run_threshold_failure():
    def fail(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_threshold(PipelineConfig(), runner=fail)


def test_run_backtest_selects_model_and_threshold(tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path), threshold_file='th.csv')
    (tmp_path / 'model_a.joblib').write_text('x')
    (tmp_path / 'model_b.joblib').write_text('x')
    pd.DataFrame({'best_threshold': [0.4, 0.6]}).to_csv(tmp_path / 'th.csv', index=False)
    captured = {}
    def fake_pipe(f_df, p_df, model, thresh):
        captured['model'] = model
        captured['thresh'] = thresh
    pipeline.run_backtest(cfg, pipeline_func=fake_pipe)
    assert captured['model'].endswith('model_b.joblib')
    assert captured['thresh'] == pytest.approx(0.5)


def test_run_backtest_pipeline_error(tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path), threshold_file='th.csv')
    def boom(*args):
        raise RuntimeError('bad')
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_backtest(cfg, pipeline_func=boom)


def test_run_report_success(monkeypatch):
    called = {}
    import src.main as src_main
    monkeypatch.setattr(src_main, 'run_pipeline_stage', lambda s: called.setdefault('stage', s))
    pipeline.run_report(PipelineConfig())
    assert called['stage'] == 'report'


def test_run_report_failure(monkeypatch):
    import src.main as src_main
    monkeypatch.setattr(src_main, 'run_pipeline_stage', lambda s: (_ for _ in ()).throw(Exception('x')))
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_report(PipelineConfig())


def test_run_all_sequence(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(pipeline, 'run_preprocess', lambda c: order.append('preprocess'))
    monkeypatch.setattr(pipeline, 'run_sweep', lambda c: order.append('sweep'))
    monkeypatch.setattr(pipeline, 'run_threshold', lambda c: order.append('threshold'))
    monkeypatch.setattr(pipeline, 'run_backtest', lambda c: order.append('backtest'))
    monkeypatch.setattr(pipeline, 'run_report', lambda c: order.append('report'))
    cfg = PipelineConfig(model_dir=str(tmp_path))
    pipeline.run_all(cfg)
    assert order == ['preprocess', 'sweep', 'threshold', 'backtest', 'report']
    assert (tmp_path / '.qa_pipeline.log').exists()


def test_main_handles_pipeline_error(monkeypatch):
    monkeypatch.setattr(pipeline, 'run_backtest', lambda c: (_ for _ in ()).throw(pipeline.PipelineError('x')))
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    res = pipeline.main(['--mode', 'backtest'])
    assert res == 1


def test_main_handles_file_not_found(monkeypatch):
    monkeypatch.setattr(pipeline, 'run_backtest', lambda c: (_ for _ in ()).throw(FileNotFoundError('x')))
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    res = pipeline.main(['--mode', 'backtest'])
    assert res == 1


def test_main_handles_value_error(monkeypatch):
    monkeypatch.setattr(pipeline, 'run_backtest', lambda c: (_ for _ in ()).throw(ValueError('x')))
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    res = pipeline.main(['--mode', 'backtest'])
    assert res == 1


def test_main_success(monkeypatch):
    monkeypatch.setattr(pipeline, 'run_preprocess', lambda c: None)
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    res = pipeline.main(['--mode', 'preprocess', '--log-level', 'info'])
    assert res == 0


def test_main_debug_sets_sample_size(monkeypatch):
    import src.main as src_main
    import src.strategy as strategy
    src_main.sample_size = 0
    strategy.sample_size = 0
    monkeypatch.setattr(pipeline, 'run_all', lambda c: None)
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    monkeypatch.setattr(pipeline, 'load_config', lambda p: PipelineConfig())
    pipeline.main(['--debug'])
    assert src_main.sample_size == 2000
    assert strategy.sample_size == 2000


def test_main_rows_override(monkeypatch):
    import src.main as src_main
    import src.strategy as strategy
    src_main.sample_size = 0
    strategy.sample_size = 0
    monkeypatch.setattr(pipeline, 'run_all', lambda c: None)
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    monkeypatch.setattr(pipeline, 'load_config', lambda p: PipelineConfig())
    pipeline.main(['--rows', '123'])
    assert src_main.sample_size == 123
    assert strategy.sample_size == 123


def test_main_live_loop_called(monkeypatch):
    called = {}
    monkeypatch.setattr(pipeline, 'run_backtest', lambda c: None)
    monkeypatch.setattr(pipeline, 'run_all', lambda c: None)
    monkeypatch.setattr(pipeline, 'setup_logging', lambda level: None)
    monkeypatch.setattr(pipeline, 'load_config', lambda p: PipelineConfig())
    monkeypatch.setattr(pipeline, 'has_gpu', lambda: False)
    import src.main as src_main
    monkeypatch.setattr(src_main, 'run_live_trading_loop', lambda n: called.setdefault('loop', n))
    pipeline.main(['--mode', 'backtest', '--live-loop', '3'])
    assert called.get('loop') == 3
