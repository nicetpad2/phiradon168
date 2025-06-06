import os
import sys
import types
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.signal_classifier as sc


def test_train_signal_classifier_no_label():
    df = pd.DataFrame({'price_change':[0], 'atr':[0], 'rsi':[0],
                       'volume_change':[0], 'ma_fast':[0], 'ma_slow':[0], 'ma_cross':[0]})
    with pytest.raises(KeyError):
        sc.train_signal_classifier(df)


def test_shap_feature_analysis_with_shap(monkeypatch):
    class DummyTreeExplainer:
        def __init__(self, model):
            pass
        def shap_values(self, X):
            return [np.zeros(len(X)), np.arange(len(X))]

    dummy_shap = types.SimpleNamespace(TreeExplainer=DummyTreeExplainer)
    monkeypatch.setattr(sc, 'shap', dummy_shap, raising=False)

    df = pd.DataFrame({'High':[2,3,4], 'Low':[1,1,2], 'Close':[1,2,3], 'Volume':[10]*3})
    df = sc.add_basic_features(df)
    df['label'] = [0,1,0]
    model, X_val, _, _ = sc.train_signal_classifier(df)
    values = sc.shap_feature_analysis(model, X_val)
    assert isinstance(values, np.ndarray)
    assert values.tolist() == list(np.arange(len(X_val)))


def test_tune_threshold_optuna_with_module(monkeypatch):
    class DummyTrial:
        def suggest_float(self, name, low, high):
            return 0.6

    class DummyStudy:
        def __init__(self):
            self.best_params = {'threshold': 0.6}
        def optimize(self, objective, n_trials):
            objective(DummyTrial())

    dummy_optuna = types.SimpleNamespace(create_study=lambda direction: DummyStudy())
    monkeypatch.setitem(sys.modules, 'optuna', dummy_optuna)

    y_true = pd.Series([0,1,0,1])
    y_proba = np.array([0.2,0.8,0.4,0.9])
    thresh = sc.tune_threshold_optuna(y_true, y_proba, n_trials=1)
    assert thresh == 0.6


def test_train_meta_model_missing_columns():
    base = pd.DataFrame({'macro_trend':[1], 'micro_trend':[0], 'ml_signal':[0.5], 'label':[1]})
    for col in ['macro_trend', 'micro_trend', 'ml_signal', 'label']:
        df = base.drop(columns=[col])
        with pytest.raises(KeyError):
            sc.train_meta_model(df)
