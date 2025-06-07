import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src.pipeline_manager as pm


def test_pipeline_manager_run(monkeypatch):
    calls = {}

    class Dummy(pm.PipelineManager):
        def load_data(self):
            calls["load"] = True
            return "df"

        def run_wfv(self, best):
            calls["wfv"] = best

        def save_outputs(self):
            calls["save"] = True

        def qa_check(self):
            calls["qa"] = True

    def fake_sweep(df, params, patch_version):
        calls["sweep"] = (df, params, patch_version)
        return {"best": 1}

    monkeypatch.setattr("src.training.run_hyperparameter_sweep", fake_sweep)

    manager = Dummy({"a": 1}, "train")
    manager.run()

    assert calls["load"]
    assert calls["sweep"] == ("df", {"a": 1}, "v5.9.1")
    assert calls["wfv"] == {"best": 1}
    assert calls["save"]
    assert calls["qa"]
