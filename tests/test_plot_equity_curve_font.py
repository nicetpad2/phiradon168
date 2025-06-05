import importlib
import os
import sys
import pandas as pd
import matplotlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

def test_plot_equity_curve_font(tmp_path, monkeypatch):
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    matplotlib.rcParams['font.family'] = ['sans-serif']
    equity = pd.Series([100, 101], index=pd.date_range('2024-01-01', periods=2, freq='D'))
    out_dir = tmp_path
    main.plot_equity_curve(equity, 'FontTest', 100, str(out_dir), 'font_test')
    assert (out_dir / 'equity_curve_v32_font_test.png').exists()
