import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline_manager import PipelineManager


def test_pipeline_manager_run_sequence(monkeypatch):
    calls = []
    pm = PipelineManager({}, mode="all")

    monkeypatch.setattr(pm, "load_data", lambda: calls.append("load") or "df")
    monkeypatch.setattr("src.training.run_hyperparameter_sweep", lambda df, p, patch_version="": calls.append("sweep") or {})
    monkeypatch.setattr(pm, "run_wfv", lambda best: calls.append("wfv"))
    monkeypatch.setattr(pm, "save_outputs", lambda: calls.append("save"))
    monkeypatch.setattr(pm, "qa_check", lambda: calls.append("qa"))

    pm.run()

    assert calls == ["load", "sweep", "wfv", "save", "qa"]
