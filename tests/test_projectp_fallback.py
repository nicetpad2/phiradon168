import runpy
import types
import sys
import os
import json
from pathlib import Path
import pytest
import src.config as config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_missing_outputs_abort(monkeypatch, tmp_path):
    """หากไม่มีไฟล์ผลลัพธ์สคริปต์ควรหยุดการทำงาน"""
    out_dir = tmp_path / "output_default"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "DATA_DIR", Path(ROOT_DIR))
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    script_path = os.path.join(ROOT_DIR, "ProjectP.py")
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(script_path, run_name="__main__")
    assert exc.value.code == 1
    # trade_log_*.csv ไม่ควรสร้างขึ้น
    for name in ["trade_log_BUY.csv", "trade_log_SELL.csv", "trade_log_NORMAL.csv"]:
        assert not (out_dir / name).exists()
