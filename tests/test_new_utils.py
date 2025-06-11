import os
import sys
import types
import pandas as pd
import builtins
import importlib
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import utils


def test_print_qa_summary_empty_df(caplog):
    trades = pd.DataFrame()
    equity = pd.DataFrame()
    with caplog.at_level(logging.WARNING):
        res = utils.print_qa_summary(trades, equity)
    assert res["total_trades"] == 0
    assert res["winrate"] == 0.0
    assert "ไม่มีไม้ที่ถูกเทรด" in caplog.text


def test_convert_thai_datetime_invalid(caplog):
    df = pd.DataFrame({"Date": ["abc"], "Timestamp": ["xx"]})
    with caplog.at_level(logging.ERROR):
        result = utils.convert_thai_datetime(df.copy())
    assert result["timestamp"].isna().all()
    assert "ไม่สามารถแปลงปี พ.ศ." in caplog.text


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


def test_print_qa_summary_valid_df(caplog):
    trades = pd.DataFrame({"pnl": [1.0, -1.0, 3.0]})
    equity = pd.DataFrame({"equity": [100.0, 101.0, 103.0]})
    with caplog.at_level(logging.INFO):
        res = utils.print_qa_summary(trades, equity)
    assert res["total_trades"] == 3
    assert res["winrate"] == 2 / 3
    assert res["avg_pnl"] == pytest.approx((1.0 - 1.0 + 3.0) / 3)
    assert res["final_equity"] == 103.0
    assert res["max_drawdown"] <= 0
    assert "=== QA SUMMARY ===" in caplog.text


def test_convert_thai_datetime_valid(tmp_path):
    df = pd.DataFrame({"Date": ["25670101"], "Timestamp": ["12:00:00"]})
    res = utils.convert_thai_datetime(df)
    assert res["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 12:00:00")


def test_prepare_csv_auto(tmp_path):
    df = pd.DataFrame({"Date": ["25670101"], "Timestamp": ["00:00:00"], "A": [1]})
    csv_path = tmp_path / "d.csv"
    df.to_csv(csv_path, index=False)
    loaded = utils.prepare_csv_auto(str(csv_path))
    assert "timestamp" in loaded.columns


def test_convert_thai_datetime_missing_cols():
    df = pd.DataFrame({"A": [1]})
    out = utils.convert_thai_datetime(df)
    assert out.equals(df)
    assert out is not df


def test_get_resource_plan_with_gpu(monkeypatch):
    fake_psutil = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=8 * 1024 ** 3),
        cpu_count=lambda logical=True: 4,
    )
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_):
            return "FakeGPU"

    fake_torch = types.SimpleNamespace(cuda=FakeCuda)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    os.environ["DEBUG_RESOURCE"] = "1"
    try:
        plan = utils.get_resource_plan(debug=True)
    finally:
        os.environ.pop("DEBUG_RESOURCE", None)
    assert plan["device"] == "cuda"
    assert plan["gpu_name"] == "FakeGPU"
    assert os.path.exists("resource_debug.log")


def test_get_resource_plan_attribute_error(monkeypatch):
    bad_psutil = types.SimpleNamespace(
        virtual_memory=lambda: (_ for _ in ()).throw(AttributeError()),
        cpu_count=lambda logical=True: (_ for _ in ()).throw(AttributeError()),
    )
    class BadCuda:
        @staticmethod
        def is_available():
            raise AttributeError

    monkeypatch.setitem(sys.modules, "psutil", bad_psutil)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=BadCuda))
    plan = utils.get_resource_plan()
    assert plan["device"] == "cpu"
    assert plan["gpu_name"] == "Unknown"


def test_get_resource_plan_no_gpu(monkeypatch):
    ps = types.SimpleNamespace(
        virtual_memory=lambda: types.SimpleNamespace(total=1 * 1024 ** 3),
        cpu_count=lambda logical=True: 2,
    )
    torch_mod = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "psutil", ps)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    plan = utils.get_resource_plan()
    assert plan["device"] == "cpu"
    assert plan["gpu_name"] == "None"
