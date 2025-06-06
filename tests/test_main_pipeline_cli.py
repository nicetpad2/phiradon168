import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline


def test_run_all_order(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_preprocess", lambda cfg: calls.append("preprocess"))
    monkeypatch.setattr(pipeline, "run_sweep", lambda cfg: calls.append("sweep"))
    monkeypatch.setattr(pipeline, "run_threshold", lambda cfg: calls.append("threshold"))
    monkeypatch.setattr(pipeline, "run_backtest", lambda cfg: calls.append("backtest"))
    monkeypatch.setattr(pipeline, "run_report", lambda cfg: calls.append("report"))
    pipeline.main(["--mode", "all"])
    assert calls == ["preprocess", "sweep", "threshold", "backtest", "report"]


def test_profile_argument(monkeypatch):
    called = {}

    def fake_run():
        called['run'] = True

    def fake_profile(func, output):
        called['func'] = func
        called['output'] = output

    monkeypatch.setattr(pipeline, "run_backtest", lambda cfg: fake_run())
    import profile_backtest
    monkeypatch.setattr(profile_backtest, "run_profile", fake_profile)

    pipeline.main(["--mode", "backtest", "--profile", "--output-file", "out.prof"])

    called['func']()
    assert called['run']
    assert called['output'] == "out.prof"

