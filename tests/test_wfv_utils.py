import pandas as pd
from src.wfv import walk_forward_grid_search, prune_features_by_importance


def dummy_backtest(df, tp=1.0, sl=1.0):
    # very simple backtest: pnl = mean close * tp - sl
    pnl = float(df['Close'].mean() * tp - sl)
    return {"pnl": pnl, "winrate": 0.5, "maxdd": 0.1}


def test_walk_forward_grid_search_basic():
    df = pd.DataFrame({"Close": range(10)})
    grid = {"tp": [1.0, 2.0], "sl": [0.5, 1.0]}
    res = walk_forward_grid_search(df, grid, dummy_backtest, train_window=4, test_window=2, step=2)
    assert not res.empty
    assert set(res.columns) == {"start", "tp", "sl", "pnl", "winrate", "maxdd"}


def test_prune_features_by_importance():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    imp = {"a": 0.02, "b": 0.005, "c": 0.5}
    new_df, dropped = prune_features_by_importance(df, imp, threshold=0.01)
    assert list(new_df.columns) == ["a", "c"]
    assert dropped == ["b"]
