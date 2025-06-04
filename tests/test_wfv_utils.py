import pandas as pd
import pytest
import logging
from src.wfv import walk_forward_grid_search, prune_features_by_importance, logger


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


def test_walk_forward_grid_search_requires_sorted_index():
    df = pd.DataFrame({"Close": range(5)}, index=[4, 3, 2, 1, 0])
    grid = {"tp": [1.0], "sl": [1.0]}
    with pytest.raises(AssertionError):
        walk_forward_grid_search(df, grid, dummy_backtest, train_window=2, test_window=1, step=1)


def test_walk_forward_grid_search_multi_objective(caplog):
    df = pd.DataFrame({"Close": range(10)})

    def bt(df, tp=1.0, sl=1.0):
        pnl = float(df["Close"].mean() * tp - sl)
        maxdd = 0.1 if sl > 0.5 else 0.05
        return {"pnl": pnl, "winrate": 0.5, "maxdd": maxdd}

    grid = {"tp": [1.0, 1.5], "sl": [1.0, 0.5]}
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO):
        res = walk_forward_grid_search(
            df,
            grid,
            bt,
            train_window=4,
            test_window=2,
            step=2,
            objective_metrics=["pnl", "maxdd"],
        )
    logger.setLevel(logging.WARNING)
    assert any("Fold 1" in m for m in caplog.messages)
    assert res.iloc[0]["tp"] == 1.5 and res.iloc[0]["sl"] == 0.5


def test_walk_forward_minimizes_dd_substrings():
    df = pd.DataFrame({"Close": range(10)})

    def bt(df, tp=1.0, sl=1.0):
        pnl = float(df["Close"].mean() * tp - sl)
        equity_dd = 0.2 if sl > 0.5 else 0.1
        return {"pnl": pnl, "winrate": 0.5, "equity_dd": equity_dd}

    grid = {"tp": [1.0, 1.5], "sl": [1.0, 0.4]}
    res = walk_forward_grid_search(
        df,
        grid,
        bt,
        train_window=4,
        test_window=2,
        step=2,
        objective_metrics=["pnl", "equity_dd"],
    )
    assert res.iloc[0]["tp"] == 1.5 and res.iloc[0]["sl"] == 0.4


def test_prune_features_by_importance():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    imp = {"a": 0.02, "b": 0.005, "c": 0.5}
    new_df, dropped = prune_features_by_importance(df, imp, threshold=0.01)
    assert list(new_df.columns) == ["a", "c"]
    assert dropped == ["b"]
