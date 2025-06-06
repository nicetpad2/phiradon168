import pandas as pd
import pytest
from src.realtime_dashboard import (
    compute_drawdown,
    compute_equity_curve,
    check_drawdown_alert,
)


def test_compute_equity_curve():
    df = pd.DataFrame({'pnl': [1, -0.5, 2]})
    equity = compute_equity_curve(df)
    assert list(equity) == [1, 0.5, 2.5]


def test_compute_drawdown():
    equity = pd.Series([1, 2, 1.5])
    dd = compute_drawdown(equity)
    assert pytest.approx(dd.tolist()) == [0.0, 0.0, -0.25]


def test_check_drawdown_alert():
    dd = pd.Series([0.0, -0.03, -0.06])
    assert check_drawdown_alert(dd, threshold=0.05) is True
    assert check_drawdown_alert(pd.Series(), threshold=0.05) is False
