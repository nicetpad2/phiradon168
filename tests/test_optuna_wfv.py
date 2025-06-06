import os
import sys
import types
import importlib.util
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_wfv():
    spec = importlib.util.spec_from_file_location(
        "src.wfv", os.path.join(ROOT_DIR, "src", "wfv.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.wfv"] = module
    spec.loader.exec_module(module)
    return module

import optuna

_real_config = importlib.import_module("src.config")
config = types.SimpleNamespace(optuna=optuna)
sys.modules["src.config"] = config


def dummy_backtest(df, signal=1.0, loss_thresh=4, atr_mult=1.0, ma_period=20):
    pnl = float(df['Close'].mean() * signal - loss_thresh)
    r_mult = pnl / loss_thresh if loss_thresh else 0.0
    return {'r_multiple': r_mult, 'winrate': 0.6, 'mdd': 0.05}


def test_optuna_walk_forward_basic():
    wfv = load_wfv()
    df = pd.DataFrame({'Close': range(12)}, index=pd.RangeIndex(12))
    space = {'signal': (0.5, 1.0, 0.5)}
    res = wfv.optuna_walk_forward(df, space, dummy_backtest, train_window=4, test_window=2, step=2, n_trials=1)
    assert 'signal' in res.columns
    assert 'value' in res.columns


def test_optuna_walk_forward_no_optuna(monkeypatch):
    wfv = load_wfv()
    df = pd.DataFrame({'Close': range(12)}, index=pd.RangeIndex(12))
    space = {'signal': (0.5, 1.0, 0.5)}
    monkeypatch.setattr(config, 'optuna', None, raising=False)
    res = wfv.optuna_walk_forward(df, space, dummy_backtest, train_window=4, test_window=2, step=2, n_trials=1)
    assert res.empty


def test_optuna_walk_forward_per_fold_basic():
    wfv = load_wfv()
    df = pd.DataFrame({'Close': range(12)}, index=pd.RangeIndex(12))
    space = {'signal': (0.5, 1.0, 0.5)}
    res = wfv.optuna_walk_forward_per_fold(df, space, dummy_backtest, train_window=4, test_window=2, step=2, n_trials=1)
    assert not res.empty
    assert 'fold' in res.columns


def test_optuna_walk_forward_per_fold_overlap_error():
    wfv = load_wfv()
    df = pd.DataFrame({'Close': range(6)}, index=[0, 2, 1, 3, 4, 5])
    space = {'signal': (0.5, 1.0, 0.5)}
    with pytest.raises(AssertionError):
        wfv.optuna_walk_forward_per_fold(df, space, dummy_backtest, train_window=4, test_window=1, step=1, n_trials=1)


def teardown_module(module):
    sys.modules["src.config"] = _real_config
