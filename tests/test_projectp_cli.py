import os
import ProjectP as proj
import pytest

def test_parse_args_modes():
    assert proj.parse_args(["--mode", "sweep"]).mode == "sweep"
    assert proj.parse_args([]).mode == "preprocess"


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
