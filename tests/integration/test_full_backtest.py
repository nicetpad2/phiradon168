import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import metrics


def test_regression_metrics():
    pnls = [1.0, -0.5, 0.2]
    res = metrics.calculate_metrics(pnls)
    assert res == {'r_multiple': 0.7, 'winrate': 2/3}
