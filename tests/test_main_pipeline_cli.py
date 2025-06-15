import os
import sys

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



def test_run_all_order(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(pipeline, "run_preprocess", lambda c: order.append("preprocess"))
    monkeypatch.setattr(pipeline, "run_sweep", lambda c: order.append("sweep"))
    monkeypatch.setattr(pipeline, "run_threshold", lambda c: order.append("threshold"))
    monkeypatch.setattr(pipeline, "run_backtest", lambda c: order.append("backtest"))
    monkeypatch.setattr(pipeline, "run_report", lambda c: order.append("report"))
    monkeypatch.setattr(pipeline, "setup_logging", lambda level: None)
    monkeypatch.setattr(pipeline, "load_config", lambda p: PipelineConfig(model_dir=str(tmp_path)))
    pipeline.main(["--mode", "all"])
    assert order == ["preprocess", "sweep", "threshold", "backtest", "report"]
    assert (tmp_path / ".qa_pipeline.log").exists()


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

