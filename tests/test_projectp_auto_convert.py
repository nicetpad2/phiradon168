import os
import sys
import runpy
import pandas as pd
import pytest


def test_auto_convert_default_dir(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Time": ["00:00:00"],
        "Open": [1.0],
        "High": [1.0],
        "Low": [1.0],
        "Close": [1.0],
    })
    df.to_csv(src_dir / "XAUUSD_M1.csv", index=False)

    monkeypatch.setenv("SOURCE_CSV_DIR", str(src_dir))
    monkeypatch.delenv("DEST_CSV_DIR", raising=False)

    import src.main as main_mod
    monkeypatch.setattr(main_mod, "setup_output_directory", lambda base, name: str(tmp_path / "out"))

    monkeypatch.setattr(sys, "argv", ["ProjectP.py", "--auto-convert"])

    with pytest.raises(SystemExit):
        runpy.run_path("ProjectP.py", run_name="__main__")

    out_file = tmp_path / "out" / "converted_csvs" / "XAUUSD_M1_thai.csv"
    assert out_file.exists()
