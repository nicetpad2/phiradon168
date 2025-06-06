import pandas as pd
import pytest

import importlib.util
import os
import sys

def load_wfv():
    spec = importlib.util.spec_from_file_location(
        "src.wfv", os.path.join(os.path.dirname(__file__), "..", "src", "wfv.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.wfv"] = module
    spec.loader.exec_module(module)
    return module


def test_is_minimize_metric_variants():
    wfv = load_wfv()
    assert wfv._is_minimize_metric("maxdd")
    assert wfv._is_minimize_metric("EQUITY_DD")
    assert not wfv._is_minimize_metric("pnl")


def test_dominates_basic_logic():
    wfv = load_wfv()
    a = {"pnl": 10.0, "maxdd": 0.1}
    b = {"pnl": 8.0, "maxdd": 0.2}
    metrics = ["pnl", "maxdd"]
    assert wfv._dominates(a, b, metrics)
    assert not wfv._dominates(b, a, metrics)


def test_grid_search_fallback_first_candidate(monkeypatch):
    wfv = load_wfv()
    df = pd.DataFrame({"Close": range(10)})
    grid = {"tp": [1, 2], "sl": [1, 2]}

    def bt(_df, tp=1, sl=1):
        return {"pnl": 1.0, "winrate": 0.5, "maxdd": 0.1}

    monkeypatch.setattr(wfv, "_dominates", lambda *args, **kwargs: True)
    res = wfv.walk_forward_grid_search(df, grid, bt, train_window=4, test_window=2, step=2)
    assert (res[["tp", "sl"]] == 1).all().all()


def test_prune_features_missing_columns():
    wfv = load_wfv()
    df = pd.DataFrame({"a": [1], "c": [3]})
    imp = {"a": 0.005, "b": 0.001, "c": 0.5}
    new_df, dropped = wfv.prune_features_by_importance(df, imp, threshold=0.01)
    assert list(new_df.columns) == ["c"]
    assert dropped == ["a"]
