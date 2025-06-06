import pandas as pd
import pytest

from src import wfv
from src.wfv import _is_minimize_metric, _dominates, walk_forward_grid_search, prune_features_by_importance


def test_is_minimize_metric_variants():
    assert _is_minimize_metric("maxdd")
    assert _is_minimize_metric("EQUITY_DD")
    assert not _is_minimize_metric("pnl")


def test_dominates_basic_logic():
    a = {"pnl": 10.0, "maxdd": 0.1}
    b = {"pnl": 8.0, "maxdd": 0.2}
    metrics = ["pnl", "maxdd"]
    assert _dominates(a, b, metrics)
    assert not _dominates(b, a, metrics)


def test_grid_search_fallback_first_candidate(monkeypatch):
    df = pd.DataFrame({"Close": range(10)})
    grid = {"tp": [1, 2], "sl": [1, 2]}

    def bt(_df, tp=1, sl=1):
        return {"pnl": 1.0, "winrate": 0.5, "maxdd": 0.1}

    monkeypatch.setattr(wfv, "_dominates", lambda *args, **kwargs: True)
    res = walk_forward_grid_search(df, grid, bt, train_window=4, test_window=2, step=2)
    assert (res[["tp", "sl"]] == 1).all().all()


def test_prune_features_missing_columns():
    df = pd.DataFrame({"a": [1], "c": [3]})
    imp = {"a": 0.005, "b": 0.001, "c": 0.5}
    new_df, dropped = prune_features_by_importance(df, imp, threshold=0.01)
    assert list(new_df.columns) == ["c"]
    assert dropped == ["a"]
