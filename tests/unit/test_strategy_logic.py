import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import metrics, drift_observer, trend_filter


def test_calculate_metrics_basic():
    res = metrics.calculate_metrics([1.0, -0.5, 0.5])
    assert res['r_multiple'] == 1.0
    assert res['winrate'] == pytest.approx(2/3)


def test_calculate_metrics_empty():
    assert metrics.calculate_metrics([]) == {'r_multiple': 0.0, 'winrate': 0.0}


def test_drift_observer_init_valid():
    obs = drift_observer.DriftObserver(['a', 'b'])
    assert obs.features == ['a', 'b'] and obs.results == {}


def test_drift_observer_init_invalid():
    with pytest.raises(ValueError):
        drift_observer.DriftObserver('not_list')


def test_apply_trend_filter_type_error():
    with pytest.raises(TypeError):
        trend_filter.apply_trend_filter(None)
