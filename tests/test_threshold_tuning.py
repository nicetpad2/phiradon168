import json
import numpy as np
import pandas as pd
import pytest
from src.evaluation import find_best_threshold
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
        n = len(X)
        p = np.linspace(0.1, 0.8, n)
        return np.column_stack([1 - p, p])

class DummyPool:
    def __init__(self, X, label=None, cat_features=None):
        self.data = X

def test_find_best_threshold():
    proba = np.array([0.1, 0.4, 0.6, 0.8])
    y = np.array([0, 0, 1, 1])
    res = find_best_threshold(proba, y)
    assert res["best_threshold"] == pytest.approx(0.4)
    assert res["best_f1"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)


def test_threshold_tuning_called(tmp_path, monkeypatch):
    trade_log = pd.DataFrame({
        'entry_time': pd.date_range('2023-01-01', periods=6, freq='min'),
        'exit_reason': ['TP','SL','TP','SL','TP','SL']
    })
    m1 = pd.DataFrame({
        'Open': np.arange(6)+1,
        'High': np.arange(6)+1.1,
        'Low': np.arange(6)+0.9,
        'Close': np.arange(6)+1,
        'ATR_14': np.ones(6)
    }, index=pd.date_range('2023-01-01', periods=6, freq='min'))
    m1_path = tmp_path / 'm1.csv'
    m1.to_csv(m1_path)

    called = {}
    def fake_find(proba, y):
        called['hit'] = True
        return {
            "best_threshold": 0.5,
            "best_f1": 0.5,
            "precision": 0.5,
            "recall": 0.5,
        }

    monkeypatch.setattr(strategy, 'find_best_threshold', fake_find)
    monkeypatch.setattr(strategy, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(strategy, 'safe_load_csv_auto', lambda p, **k: pd.read_csv(p, index_col=0), raising=False)
    from src import data_loader
    monkeypatch.setattr(data_loader, 'validate_m1_data_path', lambda p: True)
    monkeypatch.setattr(strategy, 'CatBoostClassifier', DummyCat)
    monkeypatch.setattr(strategy, 'Pool', DummyPool)
    monkeypatch.setattr(strategy, 'shap', None, raising=False)
    monkeypatch.setattr(strategy, 'check_model_overfit', lambda *a, **k: None)
    monkeypatch.setattr(strategy, 'joblib_dump', lambda obj, path: open(path, 'wb').write(b'd'))

    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with open(out_dir / 'features_main.json', 'w', encoding='utf-8') as f:
        json.dump(['Open', 'High', 'Low', 'Close', 'ATR_14'], f)

    strategy.train_and_export_meta_model(
        trade_log_df_override=trade_log,
        m1_data_path=str(m1_path),
        output_dir=str(out_dir),
        enable_dynamic_feature_selection=False,
        enable_optuna_tuning=False,
        model_type_to_train='catboost',
        enable_threshold_tuning=True
    )

    assert called.get('hit', False)
