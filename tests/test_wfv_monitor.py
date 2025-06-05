import os, sys
import pandas as pd
from src.wfv_monitor import walk_forward_validate


def backtest_pass(train, test):
    return {"pnl": 100.0, "winrate": 0.7, "maxdd": 0.05, "auc": 0.8}


def backtest_fail(train, test):
    return {"pnl": 10.0, "winrate": 0.4, "maxdd": 0.2, "auc": 0.5}


def test_walk_forward_validate_pass():
    df = pd.DataFrame({"Close": range(20)}, index=pd.RangeIndex(20))
    kpi = {"profit": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}
    res = walk_forward_validate(df, backtest_pass, kpi, n_splits=5)
    assert not res["failed"].any()


def test_walk_forward_validate_fail(monkeypatch):
    df = pd.DataFrame({"Close": range(20)}, index=pd.RangeIndex(20))
    kpi = {"profit": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}
    called = {"n": 0}

    def dummy_retrain(fold, metrics):
        called["n"] += 1

    res = walk_forward_validate(df, backtest_fail, kpi, n_splits=5, retrain_func=dummy_retrain)
    assert res["failed"].all()
    assert called["n"] == 5
