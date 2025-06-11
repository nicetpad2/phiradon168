import runpy
import types
import sys
import os
from pathlib import Path
import pandas as pd
import pytest
import src.config as config
import ProjectP

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_insufficient_rows_logs_warning(monkeypatch, tmp_path, caplog):
    out_dir = tmp_path / "output_default"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(config, "DATA_DIR", Path(ROOT_DIR))
    out_dir.mkdir()
    (out_dir / "features_main.json").write_text("[]")
    trade_file = out_dir / "trade_log_test.csv"
    trade_file.write_text("")

    monkeypatch.setattr(
        "src.evaluation.auto_train_meta_classifiers", lambda *a, **k: None, raising=False
    )
    orig_read_csv = ProjectP.pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        p = Path(path)
        if p == trade_file or p.name == "trade_log_v32_walkforward.csv.gz":
            return pd.DataFrame()
        return orig_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(ProjectP.pd, "read_csv", fake_read_csv)
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    script_path = os.path.join(ROOT_DIR, "ProjectP.py")
    with pytest.raises(ValueError):
        runpy.run_path(script_path, run_name="__main__")
