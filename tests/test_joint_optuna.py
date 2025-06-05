import pandas as pd
from sklearn.linear_model import LogisticRegression
import src.config as config
from tuning.joint_optuna import joint_optuna_optimization


def test_joint_optuna_skip(monkeypatch):
    X = pd.DataFrame({'a': range(8), 'b': range(8, 0, -1)})
    y = pd.Series([0, 1] * 4)
    monkeypatch.setattr(config, 'optuna', None, raising=False)
    best_val, best_params = joint_optuna_optimization(
        X,
        y,
        LogisticRegression,
        {'C': (0.1, 1.0, 0.1)},
        {'threshold': (0.4, 0.6, 0.1)},
        n_splits=2,
        n_trials=1,
    )
    assert best_val == 0.0
    assert best_params == {}


def test_joint_optuna_basic():
    X = pd.DataFrame({'a': range(8), 'b': range(8, 0, -1)})
    y = pd.Series([0, 1] * 4)
    best_val, best_params = joint_optuna_optimization(
        X,
        y,
        LogisticRegression,
        {'C': (0.1, 1.0, 0.1)},
        {'threshold': (0.4, 0.6, 0.1)},
        n_splits=2,
        n_trials=1,
    )
    assert isinstance(best_val, float)
    assert isinstance(best_params, dict)

