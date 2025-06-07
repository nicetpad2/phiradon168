import runpy
import types
import sys
import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_fallback_files_created(monkeypatch, tmp_path):
    """Script should generate dummy files when outputs are missing."""
    out_dir = tmp_path / "output_default"
    monkeypatch.chdir(tmp_path)
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, "src.main", types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, "argv", ["ProjectP.py"])
    script_path = os.path.join(ROOT_DIR, "ProjectP.py")
    runpy.run_path(script_path, run_name="__main__")
    files = [
        "features_main.json",
        "trade_log_BUY.csv",
        "trade_log_SELL.csv",
        "trade_log_NORMAL.csv",
    ]
    for name in files:
        fpath = out_dir / name
        assert fpath.exists()
        if name.endswith('.csv'):
            with open(fpath, encoding='utf-8') as fh:
                header = fh.readline().strip().split(',')
            # Fallback files may be empty; accept empty header
            assert header in ([
                "timestamp",
                "symbol",
                "side",
                "price",
                "size",
                "order_type",
                "status",
            ], [""])
    with open(out_dir / "features_main.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
