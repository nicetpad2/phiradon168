import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.evaluation import walk_forward_yearly_validation, detect_overfit_wfv
from src.dashboard import plot_wfv_summary


def dummy_backtest(df):
    return {"pnl": float(df["Close"].mean()), "winrate": 0.6, "maxdd": 0.1}


def test_walk_forward_yearly_validation_basic():
    dates = pd.date_range("2018-01-01", periods=5 * 365, freq="D")
    df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
    res = walk_forward_yearly_validation(df, dummy_backtest, train_years=3, test_years=1)
    assert not res.empty
    assert {"train_pnl", "test_pnl", "test_maxdd"}.issubset(res.columns)


def test_detect_overfit_wfv():
    res = pd.DataFrame({"train_pnl": [100.0, 120.0], "test_pnl": [-5.0, -10.0]})
    assert detect_overfit_wfv(res)


def test_plot_wfv_summary(tmp_path):
    df = pd.DataFrame({"fold": [1], "test_pnl": [1.0], "train_pnl": [2.0]})
    fig = plot_wfv_summary(df)
    out = tmp_path / "summary.html"
    fig.write_html(str(out))
    assert out.exists()
