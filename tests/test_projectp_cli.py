import os
import ProjectP as proj
import pytest

def test_parse_args_modes():
    assert proj.parse_args(["--mode", "sweep"]).mode == "sweep"
    assert proj.parse_args([]).mode == "preprocess"
    assert proj.parse_args(["--mode", "hyper_sweep"]).mode == "hyper_sweep"
    assert proj.parse_args(["--mode", "wfv"]).mode == "wfv"


def test_run_mode_invalid():
    with pytest.raises(ValueError):
        proj.run_mode("unknown")


def test_run_sweep_uses_absolute_path(monkeypatch):
    """run_sweep should build an absolute path to the sweep script."""
    called = {}

    def fake_run(cmd, check):
        called['path'] = cmd[1]

    monkeypatch.setattr(proj.subprocess, 'run', fake_run)
    proj.run_sweep()
    expected = os.path.join(os.path.dirname(os.path.abspath(proj.__file__)),
                            'tuning', 'hyperparameter_sweep.py')
    assert called['path'] == expected
    assert os.path.isabs(called['path'])


def test_run_threshold_uses_absolute_path(monkeypatch):
    """run_threshold should build an absolute path to the threshold script."""
    called = {}

    def fake_run(cmd, check):
        called['path'] = cmd[1]

    monkeypatch.setattr(proj.subprocess, 'run', fake_run)
    proj.run_threshold()
    expected = os.path.join(os.path.dirname(os.path.abspath(proj.__file__)),
                            'threshold_optimization.py')
    assert called['path'] == expected
    assert os.path.isabs(called['path'])


def test_run_hyperparameter_sweep_calls_module(monkeypatch):
    captured = {}

    def fake_sweep(out_dir, params, seed=0, resume=True):
        captured['params'] = params

    import importlib
    module = importlib.import_module('tuning.hyperparameter_sweep')
    monkeypatch.setattr(module, 'run_sweep', fake_sweep)
    proj.run_hyperparameter_sweep({'lr': [0.1]})
    assert captured['params'] == {'lr': [0.1]}


def test_run_threshold_optimization_calls_module(monkeypatch):
    called = {}

    def fake_opt():
        called['ok'] = True
        return 'df'

    import importlib
    module = importlib.import_module('threshold_optimization')
    monkeypatch.setattr(module, 'run_threshold_optimization', fake_opt)
    result = proj.run_threshold_optimization()
    assert called.get('ok') is True
    assert result == 'df'


def test_run_full_pipeline_sequence(monkeypatch):
    calls = []

    monkeypatch.setattr(proj, 'run_preprocess', lambda: calls.append('pre'))
    monkeypatch.setattr(
        proj, 'run_hyperparameter_sweep', lambda params: calls.append('sweep')
    )
    monkeypatch.setattr(
        proj, 'run_threshold_optimization', lambda: calls.append('th')
    )
    monkeypatch.setattr(proj, 'run_backtest', lambda: calls.append('back'))
    monkeypatch.setattr(proj, 'run_report', lambda: calls.append('rep'))
    proj.run_full_pipeline()
    assert calls == ['pre', 'sweep', 'th', 'back', 'rep']


def test_run_mode_wfv(monkeypatch):
    called = {}
    monkeypatch.setattr(proj, 'run_walkforward', lambda: called.setdefault('run', True))
    proj.run_mode('wfv')
    assert called.get('run') is True


def test_run_mode_all(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(proj, 'run_hyperparameter_sweep', lambda params: calls.append('sweep'))
    best = tmp_path / 'best_params.json'
    best.write_text('{"MIN_SIGNAL_SCORE_ENTRY": 0.9}')
    monkeypatch.setattr(proj.os.path, 'join', lambda *a: str(best))
    monkeypatch.setattr(proj.os.path, 'exists', lambda p: True)
    monkeypatch.setattr(proj, 'update_config_from_dict', lambda d: calls.append('update'))
    monkeypatch.setattr(proj, 'run_walkforward', lambda: calls.append('wfv'))
    proj.run_mode('all')
    assert calls == ['sweep', 'update', 'wfv']
