import importlib
import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_plot_equity_curve_import_and_run(tmp_path):
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert hasattr(main, 'plot_equity_curve')
    equity = pd.Series([100, 105], index=pd.date_range("2024-01-01", periods=2, freq="D"))
    out_dir = tmp_path
    main.plot_equity_curve(equity, "Test", 100, str(out_dir), "test")
    files = list(out_dir.glob("equity_curve_v32_test.png"))
    assert len(files) == 1
