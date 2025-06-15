import numpy as np
from src.evaluation import sortino_ratio, calmar_ratio


def test_sortino_ratio_basic():
    returns = [0.1, -0.05, 0.07, -0.02]
    ratio = sortino_ratio(returns)
    assert ratio > 0


def test_calmar_ratio_basic():
    equity = np.array([100, 105, 102, 110])
    ratio = calmar_ratio(equity)
    assert ratio > 0
