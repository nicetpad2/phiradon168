import runpy
import pytest
import pandas as pd
import threshold_optimization as to


def test_parse_args_defaults():
    args = to.parse_args([])
    assert args.trials == 10
    assert args.study_name == "threshold-wfv"
    assert args.direction == "maximize"
    assert args.output_dir == "models"


def test_parse_args_custom():
    args = to.parse_args([
        "--output_dir", "out",
        "--trials", "5",
        "--study-name", "abc",
        "--direction", "minimize",
        "--timeout", "60",
    ])
    assert args.output_dir == "out"
    assert args.trials == 5
    assert args.study_name == "abc"
    assert args.direction == "minimize"
    assert args.timeout == 60


def test_run_threshold_optimization(tmp_path):
    import numpy as np

    state = np.random.get_state()
    try:
        df = to.run_threshold_optimization(
            output_dir=str(tmp_path), trials=1, timeout=1
        )
    finally:
        np.random.set_state(state)

    assert set(df.columns) == {"best_threshold", "best_value"}
    assert 0.0 <= df["best_threshold"].iloc[0] <= 1.0
    assert (tmp_path / "threshold_wfv_optuna_results.csv").exists()
    assert (tmp_path / "threshold_wfv_optuna_results.json").exists()


def test_run_threshold_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(to, "optuna", None)
    warnings = []
    monkeypatch.setattr(to.logger, "warning", lambda msg: warnings.append(msg))
    df = to.run_threshold_optimization(output_dir=str(tmp_path), trials=1)
    assert df["best_threshold"].iloc[0] == 0.5
    assert df["best_value"].iloc[0] == 0.0
    assert warnings and "optuna not available" in warnings[0]
    assert (tmp_path / "threshold_wfv_optuna_results.csv").exists()
    assert (tmp_path / "threshold_wfv_optuna_results.json").exists()


def test_main_invokes_run(monkeypatch, tmp_path):
    captured = {}

    def fake_run(output_dir, trials, study_name, direction, timeout):
        captured.update(
            dict(
                output_dir=output_dir,
                trials=trials,
                study_name=study_name,
                direction=direction,
                timeout=timeout,
            )
        )
        return pd.DataFrame({"best_threshold": [0.5], "best_value": [0.0]})

    monkeypatch.setattr(to, "run_threshold_optimization", fake_run)
    code = to.main([
        "--output_dir", str(tmp_path),
        "--trials", "3",
        "--study-name", "s",
        "--direction", "maximize",
        "--timeout", "2",
    ])
    assert code == 0
    assert captured == {
        "output_dir": str(tmp_path),
        "trials": 3,
        "study_name": "s",
        "direction": "maximize",
        "timeout": 2,
    }


def test_entrypoint_subprocess(tmp_path):
    import sys
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "threshold_optimization.py",
            "--output_dir",
            str(tmp_path),
            "--trials",
            "1",
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "threshold_wfv_optuna_results.csv").exists()
