import builtins
import importlib
import runpy
import types
import sys
import logging
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import src.config as config


def test_projectp_import_without_pynvml(monkeypatch):
    """Module should handle missing pynvml gracefully."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no nvml")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "ProjectP", raising=False)
    module = importlib.import_module("ProjectP")
    assert module.pynvml is None
    assert module.nvml_handle is None


def test_projectp_main_logs_gpu_status(monkeypatch, caplog):
    """Running the script should log GPU availability."""
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    with caplog.at_level(logging.INFO):
        runpy.run_path("ProjectP.py", run_name="__main__")
    assert any("GPU not available" in m for m in caplog.messages)


def test_projectp_logs_gpu_release(monkeypatch, caplog):
    """NVML handle should be shut down and logged."""
    dummy_nvml = types.SimpleNamespace()

    def fake_shutdown():
        dummy_nvml.shutdown_called = True

    dummy_nvml.nvmlInit = lambda: None
    dummy_nvml.nvmlDeviceGetHandleByIndex = lambda idx: "H"
    dummy_nvml.nvmlShutdown = fake_shutdown
    dummy_nvml.shutdown_called = False

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            return dummy_nvml
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    dummy_main = lambda: None
    main_mod = types.SimpleNamespace(
        main=dummy_main, pynvml=dummy_nvml, nvml_handle="H", USE_GPU_ACCELERATION=True
    )
    monkeypatch.setitem(sys.modules, "src.main", main_mod)
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    with caplog.at_level(logging.INFO):
        runpy.run_path("ProjectP.py", run_name="__main__")
    assert dummy_nvml.shutdown_called is True
    assert any("GPU resources released" in m for m in caplog.messages)


def test_projectp_output_audit(monkeypatch, caplog, tmp_path):
    """QA audit should log presence or absence of output files."""
    out_dir = tmp_path / "output_default"
    out_dir.mkdir()
    # Create only some files
    (out_dir / "features_main.json").write_text("{}")
    (out_dir / "trade_log_NORMAL.csv").write_text("")

    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    config.logger.setLevel(logging.INFO)
    script_path = os.path.join(ROOT_DIR, "ProjectP.py")
    with caplog.at_level(logging.INFO):
        runpy.run_path(script_path, run_name="__main__")
    # Should log present for existing file and missing for absent file
    assert any("[QA] Output present" in m for m in caplog.messages)
    assert any("[QA] Output missing" in m for m in caplog.messages)
