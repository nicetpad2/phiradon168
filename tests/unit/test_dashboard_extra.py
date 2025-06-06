import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src import dashboard


def test_create_dashboard_errors():
    s = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        dashboard.create_dashboard(pd.Series(dtype=float), s, s)
    with pytest.raises(ValueError):
        dashboard.create_dashboard(s, pd.Series(dtype=float), s)
    with pytest.raises(ValueError):
        dashboard.create_dashboard(s, s, pd.Series(dtype=float))


def test_create_and_save_dashboard(tmp_path):
    idx = pd.date_range('2024-01-01', periods=3, freq='D')
    s = pd.Series([1, 2, 3], index=idx)
    fig = dashboard.create_dashboard(s, s, s)
    out = tmp_path / 'dash.html'
    dashboard.save_dashboard(fig, str(out))
    assert out.is_file()
