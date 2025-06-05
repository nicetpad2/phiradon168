import os
import sys
import types
import pandas as pd
import builtins
import importlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import utils


def test_print_qa_summary_empty_df(capsys):
    trades = pd.DataFrame()
    equity = pd.DataFrame()
    res = utils.print_qa_summary(trades, equity)
    captured = capsys.readouterr().out
    assert res["total_trades"] == 0
    assert res["winrate"] == 0.0
    assert "ไม่มีไม้ที่ถูกเทรด" in captured


def test_convert_thai_datetime_invalid(tmp_path):
    df = pd.DataFrame({"Date": ["abc"], "Timestamp": ["xx"]})
    log_file = tmp_path / "error_log.txt"
    os.chdir(tmp_path)
    try:
        result = utils.convert_thai_datetime(df.copy())
        assert result["timestamp"].isna().all()
        assert log_file.exists()
    finally:
        os.chdir(ROOT_DIR)


def test_get_resource_plan_missing_modules(monkeypatch):
    def fake_import(name, *args, **kwargs):
        if name in {"psutil", "torch"}:
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    plan = utils.get_resource_plan()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert set(plan) == {"ram_gb", "threads", "device", "gpu_name"}
