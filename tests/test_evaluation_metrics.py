
import pandas as pd
import pytest
from src.evaluation import (
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    compute_underwater_curve,
)


def test_calculate_sortino_ratio_basic():
    returns = pd.Series([0.1, -0.05, 0.05])
    ratio = calculate_sortino_ratio(returns)
    assert ratio > 0


def test_calculate_calmar_ratio_basic():
    returns = pd.Series([0.01, -0.02, 0.03])
    calmar = calculate_calmar_ratio(returns, -0.02)
    assert calmar > 0


def test_calculate_max_drawdown_and_underwater():
    equity = pd.Series([1.0, 1.1, 0.9, 1.2])
    md = calculate_max_drawdown(equity)
    assert md == pytest.approx(-0.2)
    uw = compute_underwater_curve(equity)
    assert len(uw) == 4
    assert uw.min() < 0

