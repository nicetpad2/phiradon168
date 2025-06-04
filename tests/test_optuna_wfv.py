import os, sys
import pandas as pd
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
import src.config as config
from src.wfv import optuna_walk_forward


def dummy_backtest(df, signal=1.0, loss_thresh=4, atr_mult=1.0, ma_period=20):
    pnl = float(df['Close'].mean() * signal - loss_thresh)
    r_mult = pnl / loss_thresh if loss_thresh else 0.0
    return {'r_multiple': r_mult, 'winrate': 0.6, 'mdd': 0.05}


def test_optuna_walk_forward_basic():
    df = pd.DataFrame({'Close': range(12)}, index=pd.RangeIndex(12))
    space = {'signal': (0.5, 1.0, 0.5)}
    res = optuna_walk_forward(df, space, dummy_backtest, train_window=4, test_window=2, step=2, n_trials=1)
    assert 'signal' in res.columns
    assert 'value' in res.columns


def test_optuna_walk_forward_no_optuna(monkeypatch):
    df = pd.DataFrame({'Close': range(12)}, index=pd.RangeIndex(12))
    space = {'signal': (0.5, 1.0, 0.5)}
    monkeypatch.setattr(config, 'optuna', None, raising=False)
    res = optuna_walk_forward(df, space, dummy_backtest, train_window=4, test_window=2, step=2, n_trials=1)
    assert res.empty
