import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.signal_classifier as sc


def test_create_label_from_backtest_basic():
    df = pd.DataFrame({'High': [1,2,3,4,5,6,7,8,9,10,11,12], 'Close': [1]*12, 'Volume': range(12)})
    labeled = sc.create_label_from_backtest(df, target_point=5)
    assert 'label' in labeled.columns
    assert labeled['label'].iloc[4] == 1
    assert labeled['label'].iloc[0] == 0


def test_add_basic_features_columns_exist():
    df = pd.DataFrame({'High': [2,3,4,5,6], 'Low': [1,1,2,3,4], 'Close': [1,2,3,4,5], 'Volume': [10]*5})
    enriched = sc.add_basic_features(df)
    expected_cols = {'price_change','atr','rsi','volume_change','ma_fast','ma_slow','ma_cross'}
    assert expected_cols.issubset(enriched.columns)


def test_train_signal_classifier_returns_proba():
    df = pd.DataFrame({'High': [2,3,4,5,6,7], 'Low': [1,1,2,3,4,5], 'Close': [1,2,3,4,5,6], 'Volume': [10]*6})
    df = sc.add_basic_features(df)
    df['label'] = [0,0,0,1,1,1]
    model, X_val, y_val, proba = sc.train_signal_classifier(df)
    assert len(proba) == len(y_val)
    assert hasattr(model, 'predict')


def test_shap_feature_analysis_no_shap(monkeypatch):
    monkeypatch.setattr(sc, 'shap', None, raising=False)
    df = pd.DataFrame({'High': [2,3,4,5,6,7], 'Low': [1,1,2,3,4,5], 'Close': [1,2,3,4,5,6], 'Volume': [10]*6})
    df = sc.add_basic_features(df)
    df['label'] = [0,0,0,1,1,1]
    model, X_val, y_val, proba = sc.train_signal_classifier(df)
    assert sc.shap_feature_analysis(model, X_val) is None


def test_tune_threshold_optuna_no_optuna(monkeypatch):
    monkeypatch.setitem(sys.modules, 'optuna', None)
    y_true = pd.Series([0,1,0,1], dtype=int)
    y_proba = np.array([0.1,0.8,0.2,0.9], dtype=float)
    thresh = sc.tune_threshold_optuna(y_true, y_proba, n_trials=2)
    assert isinstance(thresh, float)


def test_train_meta_model_basic():
    df = pd.DataFrame({'macro_trend':[1,0,1], 'micro_trend':[0,1,0], 'ml_signal':[0.6,0.4,0.7], 'label':[1,0,1]})
    model = sc.train_meta_model(df)
    assert hasattr(model, 'predict')
