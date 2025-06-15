import os
import pandas as pd
import pytest

from src.utils.model_utils import get_latest_model_and_threshold


def test_get_latest_model_and_threshold(tmp_path):
    (tmp_path / 'model_1.joblib').write_text('a')
    (tmp_path / 'model_2.joblib').write_text('b')
    pd.DataFrame({'best_threshold': [0.4, 0.6]}).to_csv(tmp_path / 'th.csv', index=False)
    model, thresh = get_latest_model_and_threshold(str(tmp_path), 'th.csv')
    assert model.endswith('model_2.joblib')
    assert thresh == pytest.approx(0.5)


def test_get_latest_model_and_threshold_missing(tmp_path):
    model, thresh = get_latest_model_and_threshold(str(tmp_path), 'th.csv')
    assert model is None
    assert thresh is None


def test_get_latest_model_and_threshold_first(tmp_path):
    (tmp_path / 'model_a.joblib').write_text('x')
    pd.DataFrame({'best_threshold': [0.7]}).to_csv(tmp_path / 'th.csv', index=False)
    model, thresh = get_latest_model_and_threshold(str(tmp_path), 'th.csv', take_first=True)
    assert thresh == pytest.approx(0.7)
