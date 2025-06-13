import pandas as pd
import numpy as np
import pytest
from src.utils.model_utils import (
    set_last_training_timestamp,
    predict_with_time_check,
)


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])


def test_predict_with_time_check_pass(tmp_path):
    model = DummyModel()
    set_last_training_timestamp(model, pd.Timestamp("2024-01-01"))
    X = pd.DataFrame({"a": [1]})
    prob = predict_with_time_check(model, X, pd.Timestamp("2024-01-02"))
    assert 0.0 <= prob <= 1.0


def test_predict_with_time_check_assert():
    model = DummyModel()
    set_last_training_timestamp(model, pd.Timestamp("2024-01-02"))
    X = pd.DataFrame({"a": [1]})
    with pytest.raises(AssertionError):
        predict_with_time_check(model, X, pd.Timestamp("2024-01-01"))
