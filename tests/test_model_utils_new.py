import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.model_utils import (
    save_model,
    load_model,
    evaluate_model,
    predict,
)


def test_save_load_and_predict(tmp_path):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = [0, 1]
    model = LogisticRegression()
    model.fit(X, y)
    path = tmp_path / 'm.pkl'
    save_model(model, str(path))
    loaded = load_model(str(path))
    prob = predict(loaded, X.iloc[[0]])
    assert 0.0 <= prob <= 1.0
    acc, auc = evaluate_model(loaded, X, y)
    assert acc >= 0


def test_predict_with_class_idx(tmp_path):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = [0, 1]
    model = LogisticRegression()
    model.fit(X, y)
    path = tmp_path / 'm.pkl'
    save_model(model, str(path))
    loaded = load_model(str(path))
    p0 = predict(loaded, X.iloc[[0]], class_idx=0)
    p1 = predict(loaded, X.iloc[[0]], class_idx=1)
    assert abs(p0 + p1 - 1.0) < 1e-6


def test_evaluate_model_without_proba(caplog):
    class Dummy:
        def predict(self, X):
            return [0] * len(X)

    df = pd.DataFrame({'a': [0], 'b': [1]})
    dummy = Dummy()
    with caplog.at_level(logging.ERROR):
        result = evaluate_model(dummy, df, [0])
    assert result is None
    assert any('support predict_proba' in m for m in caplog.messages)


def test_load_model_missing_file(tmp_path, caplog):
    missing = tmp_path / 'none.pkl'
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            load_model(str(missing))
    assert any('Model file not found' in m for m in caplog.messages)
