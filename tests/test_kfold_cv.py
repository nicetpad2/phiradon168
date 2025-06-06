import logging
import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.training as training


class DummyModel(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, X, y, eval_set=None, use_best_model=True):
        return super().fit(X, y)


def test_kfold_cv_model_catboost_missing(monkeypatch, caplog):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = pd.Series([0, 1])
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    with caplog.at_level(logging.ERROR):
        res = training.kfold_cv_model(X, y, model_type='catboost', n_splits=2)
    assert res == {}
    assert any('catboost not available' in m for m in caplog.messages)


def test_kfold_cv_model_basic(monkeypatch):
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    monkeypatch.setattr(training, 'CatBoostClassifier', DummyModel, raising=False)
    res = training.kfold_cv_model(X, y, model_type='catboost', n_splits=2, early_stopping_rounds=None)
    assert 'auc' in res and 'f1' in res
