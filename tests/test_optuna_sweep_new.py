import pandas as pd
from src.training import optuna_sweep


def test_optuna_sweep_basic(tmp_path, monkeypatch):
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])
    monkeypatch.setattr('src.training.optuna', None, raising=False)
    params = optuna_sweep(X, y, n_trials=1, output_path=str(tmp_path / 'm.pkl'))
    assert params == {}
