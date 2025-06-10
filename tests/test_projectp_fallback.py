import runpy
import types
import sys
import os
import json
from pathlib import Path
import pandas as pd
import pytest
import src.config as config
import ProjectP

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_missing_outputs_creates_dummy(monkeypatch, tmp_path, caplog):
    """เมื่อไม่มีไฟล์ trade_log สคริปต์ควรสร้างไฟล์เปล่าสำหรับทำงานต่อ"""
    out_dir = tmp_path / "output_default"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(config, "DATA_DIR", Path(ROOT_DIR))
    out_dir.mkdir()
    # Pre-create features_main.json to skip heavy generation
    (out_dir / "features_main.json").write_text("[]")
    # Skip real meta-classifier training to avoid file dependency
    monkeypatch.setattr(
        "src.evaluation.auto_train_meta_classifiers",
        lambda *a, **k: None,
        raising=False,
    )
    orig_read_csv = ProjectP.pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if str(path).endswith("trade_log_v32_walkforward.csv.gz"):
            return pd.DataFrame()
        return orig_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(ProjectP.pd, "read_csv", fake_read_csv)
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    script_path = os.path.join(ROOT_DIR, "ProjectP.py")
    with caplog.at_level("WARNING"):
        runpy.run_path(script_path, run_name="__main__")
    assert (out_dir / "trade_log_dummy.csv").exists()
    assert (out_dir / "features_main.json").exists()
