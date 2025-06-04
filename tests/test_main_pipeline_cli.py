import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline


def test_run_all_order(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_preprocess", lambda: calls.append("preprocess"))
    monkeypatch.setattr(pipeline, "run_sweep", lambda: calls.append("sweep"))
    monkeypatch.setattr(pipeline, "run_threshold", lambda: calls.append("threshold"))
    monkeypatch.setattr(pipeline, "run_backtest", lambda: calls.append("backtest"))
    monkeypatch.setattr(pipeline, "run_report", lambda: calls.append("report"))
    pipeline.main(["--stage", "all"])
    assert calls == ["preprocess", "sweep", "threshold", "backtest", "report"]


def test_profile_argument(monkeypatch):
    called = {}

    def fake_run():
        called['run'] = True

    def fake_profile(func, output):
        called['func'] = func
        called['output'] = output

    monkeypatch.setattr(pipeline, "run_backtest", fake_run)
    import profile_backtest
    monkeypatch.setattr(profile_backtest, "run_profile", fake_profile)

    pipeline.main(["--stage", "backtest", "--profile", "--output-file", "out.prof"])

    assert called['func'] is fake_run
    assert called['output'] == "out.prof"

