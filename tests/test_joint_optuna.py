import pandas as pd
from sklearn.linear_model import LogisticRegression
import src.config as config
from tuning.joint_optuna import joint_optuna_optimization
import warnings


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


def test_joint_optuna_no_auc():
    X = pd.DataFrame({'a': range(6), 'b': range(6)})
    y = pd.Series([0, 1, 0, 0, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_val, best_params = joint_optuna_optimization(
            X,
            y,
            LogisticRegression,
            {'max_iter': (10, 20, 10)},
            {'threshold': (0.4, 0.6, 0.1)},
            n_splits=2,
            n_trials=1,
        )
    assert best_val == 0.0
    assert 'max_iter' in best_params and 'threshold' in best_params


def test_joint_optuna_int_params():
    X = pd.DataFrame({'a': range(6), 'b': [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 1, 0, 1, 1, 0])
    best_val, best_params = joint_optuna_optimization(
        X,
        y,
        LogisticRegression,
        {'max_iter': (10, 20, 10)},
        {'threshold': (0.4, 0.6, 0.1)},
        n_splits=2,
        n_trials=1,
    )
    assert 0.0 <= best_val <= 1.0
    assert isinstance(best_params.get('max_iter'), int)

