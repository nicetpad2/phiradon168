import logging
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure imports work when running this file directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src.training as training


def test_save_model_none(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    training.save_model(None, str(tmp_path), 'modelX')
    log_file = tmp_path / 'modelX_qa.log'
    assert log_file.exists()
    assert "No model was trained" in caplog.text


def test_real_train_func_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        training.real_train_func(output_dir=str(tmp_path), trade_log_path='no.csv', m1_path='no.csv')


def test_real_train_func_m1_empty(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'profit': [1]}).to_csv(trade_path, index=False)
    pd.DataFrame(columns=['Open']).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    with pytest.raises(ValueError):
        training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))


def test_real_train_func_no_numeric_columns(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'profit': [1]}).to_csv(trade_path, index=False)
    pd.DataFrame({'A': ['a', 'b']}).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    with pytest.raises(ValueError):
        training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))


def test_real_train_func_no_numeric_target(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'text': ['x', 'y']}).to_csv(trade_path, index=False)
    pd.DataFrame({'Open': [1, 2], 'High': [2, 3], 'Low': [0, 1], 'Close': [1.5, 2.5]}).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    with pytest.raises(ValueError):
        training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))


def test_real_train_func_pnl_column(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'pnl_usd_net': [1, -1, 2]}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [2, 3, 4],
        'Low': [0, 1, 2],
        'Close': [1.5, 2.5, 3.5]
    }).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    res = training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))
    assert 'model_path' in res


def test_real_train_func_auc_calculated(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'profit': [1, -1, 2, -2, 3, -3, 4, -4]}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': range(8),
        'High': range(1, 9),
        'Low': range(-1, 7),
        'Close': range(2, 10)
    }).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    res = training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))
    assert not np.isnan(res['metrics']['auc'])


def test_real_train_func_other_numeric_target(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'other': [1, 0, 1]}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [2, 3, 4],
        'Low': [0, 1, 2],
        'Close': [1.5, 2.5, 3.5]
    }).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    res = training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))
    assert 'model_path' in res


def test_time_series_cv_auc():
    X = pd.DataFrame({'a': [0, 1, 0, 1, 0, 1], 'b': [1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 0, 0, 0, 0, 0])
    auc1 = training._time_series_cv_auc(training.RandomForestClassifier, X, y, n_splits=3)
    assert auc1 == 0.5
    y2 = pd.Series([0, 1, 0, 1, 0, 1])
    auc2 = training._time_series_cv_auc(training.RandomForestClassifier, X, y2, n_splits=3)
    assert 0.0 <= auc2 <= 1.0


class OverfitModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.next_is_train = True
    def fit(self, X, y, eval_set=None, use_best_model=True):
        self.y_train = y
    def predict_proba(self, X):
        if self.next_is_train:
            self.next_is_train = False
            return np.column_stack([1 - self.y_train, self.y_train])
        return np.tile([0.6, 0.4], (len(X), 1))


def test_kfold_cv_model_overfit(monkeypatch, caplog):
    X = pd.DataFrame({'a': [0, 1, 0, 1, 0, 1, 0, 1], 'b': [1, 0, 1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    monkeypatch.setattr(training, 'CatBoostClassifier', OverfitModel, raising=False)
    with caplog.at_level(logging.WARNING):
        res = training.kfold_cv_model(X, y, model_type='catboost', n_splits=2)
    assert 'auc' in res and 'f1' in res
    assert any('Overfitting detected' in m for m in caplog.messages)


def test_kfold_cv_model_rf_missing(monkeypatch, caplog):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = pd.Series([0, 1])
    monkeypatch.setattr(training, 'RandomForestClassifier', None, raising=False)
    with caplog.at_level(logging.ERROR):
        res = training.kfold_cv_model(X, y, model_type='rf')
    assert res == {}
    assert any('RandomForest not available' in m for m in caplog.messages)


def test_kfold_cv_model_extra_params(monkeypatch):
    class Dummy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.inner = training.LogisticRegression()
        def fit(self, X, y, eval_set=None, use_best_model=True):
            self.inner.fit(X, y)
        def predict_proba(self, X):
            return self.inner.predict_proba(X)
    monkeypatch.setattr(training, 'CatBoostClassifier', Dummy, raising=False)
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    res = training.kfold_cv_model(X, y, model_type='catboost', n_splits=2, early_stopping_rounds=10, l2_leaf_reg=2.0)
    assert 'auc' in res and 'f1' in res


def test_kfold_cv_model_rf_basic(monkeypatch):
    class RF:
        def __init__(self, **kwargs):
            self.inner = training.LogisticRegression()
        def fit(self, X, y):
            self.inner.fit(X, y)
        def predict_proba(self, X):
            return self.inner.predict_proba(X)
    monkeypatch.setattr(training, 'RandomForestClassifier', RF, raising=False)
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    res = training.kfold_cv_model(X, y, model_type='rf', n_splits=2)
    assert 'auc' in res and 'f1' in res


def test_optuna_sweep_success(tmp_path, monkeypatch):
    class Trial:
        def suggest_int(self, name, low, high):
            return low
    class Study:
        def __init__(self):
            self.best_params = {}
        def optimize(self, func, n_trials):
            func(Trial())
            self.best_params = {'n_estimators': 50, 'max_depth': 3}
    class Optuna:
        def create_study(self, direction='maximize'):
            return Study()
    monkeypatch.setattr(training, 'RandomForestClassifier', lambda **k: OverfitModel(), raising=False)
    monkeypatch.setattr(training, 'save_model', lambda *a, **k: None)
    monkeypatch.setattr(training, 'evaluate_model', lambda *a, **k: (1.0, 1.0))
    import src.config as config
    monkeypatch.setattr(config, 'optuna', Optuna(), raising=False)
    monkeypatch.setattr(training, 'USE_GPU_ACCELERATION', True, raising=False)
    params = training.optuna_sweep(pd.DataFrame({'a': [0, 1]}), pd.Series([0, 1]), n_trials=1, output_path=str(tmp_path/'m.pkl'))
    assert params == {'n_estimators': 50, 'max_depth': 3}


def test_train_lightgbm_mtf_no_files(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(training, 'LGBMClassifier', lambda *a, **k: None, raising=False)
    with caplog.at_level(logging.ERROR):
        res = training.train_lightgbm_mtf('missing1.csv', 'missing2.csv', str(tmp_path))
    assert res is None
    assert any('M1 or M15 data not found' in m for m in caplog.messages)


def test_train_lightgbm_mtf_low_auc(tmp_path, monkeypatch, caplog):
    timestamps = pd.date_range('2024-01-01', periods=5, freq='1min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Open': [0, 1, 0, 1, 0],
        'High': [0, 1, 0, 1, 0],
        'Low': [0, 1, 0, 1, 0],
        'Close': [0, 1, 0, 1, 0],
        'Volume': 1
    })
    df.to_csv(tmp_path/'m1.csv', index=False)
    df.iloc[::5].to_csv(tmp_path/'m15.csv', index=False)
    monkeypatch.setattr(training, '_time_series_cv_auc', lambda *a, **k: 0.6)
    monkeypatch.setattr(training, 'LGBMClassifier', lambda *a, **k: OverfitModel())
    with caplog.at_level(logging.ERROR):
        res = training.train_lightgbm_mtf(str(tmp_path/'m1.csv'), str(tmp_path/'m15.csv'), str(tmp_path))
    assert res is None
    assert any('AUC below threshold' in m for m in caplog.messages)


def test_real_train_func_single_row(tmp_path, monkeypatch, caplog):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'profit': [1]}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': [1],
        'High': [2],
        'Low': [0],
        'Close': [1.5]
    }).to_csv(m1_path, index=False)
    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)
    res = training.real_train_func(output_dir=str(tmp_path), trade_log_path=str(trade_path), m1_path=str(m1_path))
    assert res['metrics']['accuracy'] == -1.0
    assert np.isnan(res['metrics']['auc'])
