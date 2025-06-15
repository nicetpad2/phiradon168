import pandas as pd
import numpy as np
import os
import logging
from src import strategy

class DummyCat:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.feature_names_ = None
    def fit(self, X, y, cat_features=None, eval_set=None):
        self.feature_names_ = list(X.columns)
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X))*0.4, np.ones(len(X))*0.6])

class DummyPool:
    def __init__(self, X, label=None, cat_features=None):
        self.data = X


def test_train_and_export_meta_model(tmp_path, monkeypatch):
    trade_log = pd.DataFrame({
        'entry_time': pd.date_range('2023-01-01', periods=5, freq='min'),
        'exit_reason': ['TP', 'SL', 'TP', 'SL', 'TP']
    })
    m1 = pd.DataFrame({
        'Open': np.linspace(1, 5, 5),
        'High': np.linspace(1, 5, 5)+0.1,
        'Low': np.linspace(1, 5, 5)-0.1,
        'Close': np.linspace(1, 5, 5),
        'ATR_14': np.ones(5)
    }, index=pd.date_range('2023-01-01', periods=5, freq='min'))
    m1_path = tmp_path / 'm1.csv'
    m1.to_csv(m1_path)
    monkeypatch.setattr(strategy, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(strategy, 'safe_load_csv_auto', lambda p, **k: pd.read_csv(p, index_col=0), raising=False)
    monkeypatch.setattr(strategy, 'CatBoostClassifier', DummyCat)
    monkeypatch.setattr(strategy, 'Pool', DummyPool)
    monkeypatch.setattr(strategy, 'joblib_dump', lambda obj, path: open(path, 'wb').write(b'dummy'))

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    saved, feats = strategy.train_and_export_meta_model(
        trade_log_df_override=trade_log,
        m1_data_path=str(m1_path),
        output_dir=str(out_dir),
        enable_dynamic_feature_selection=False,
        enable_optuna_tuning=False,
        model_type_to_train='catboost'
    )
    assert saved is None
    assert feats == []


def test_train_and_export_meta_model_index_conversion(tmp_path, monkeypatch, caplog):
    trade_log = pd.DataFrame({
        'entry_time': pd.date_range('2024-01-01', periods=3, freq='min'),
        'exit_reason': ['TP', 'SL', 'TP']
    })
    m1 = pd.DataFrame({
        'Open': [1.0, 1.1, 1.2],
        'High': [1.1, 1.2, 1.3],
        'Low': [0.9, 1.0, 1.1],
        'Close': [1.0, 1.1, 1.2],
        'ATR_14': [0.1, 0.1, 0.1],
        'datetime': pd.date_range('2024-01-01', periods=3, freq='min')
    })
    m1_path = tmp_path / 'm1_range.csv'
    m1.to_csv(m1_path, index=False)

    monkeypatch.setattr(strategy, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(strategy, 'CatBoostClassifier', DummyCat)
    monkeypatch.setattr(strategy, 'Pool', DummyPool)
    monkeypatch.setattr(strategy, 'joblib_dump', lambda obj, path: open(path, 'wb').write(b'd'))
    monkeypatch.setattr(strategy, 'load_final_m1_data', lambda p, t: pd.read_csv(p))

    out_dir = tmp_path / 'out2'
    out_dir.mkdir()

    with caplog.at_level(logging.INFO):
        strategy.train_and_export_meta_model(
            trade_log_df_override=trade_log,
            m1_data_path=str(m1_path),
            output_dir=str(out_dir),
            enable_dynamic_feature_selection=False,
            enable_optuna_tuning=False,
            model_type_to_train='catboost'
        )
    assert 'Successfully converted index of M1 DataFrame to DatetimeIndex.' in caplog.text
