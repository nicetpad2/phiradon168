import importlib.util
import os
import sys
import types
import pandas as pd
import pytest

def load_wfv():
    spec = importlib.util.spec_from_file_location(
        "src.wfv", os.path.join(os.path.dirname(__file__), "..", "src", "wfv.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.wfv"] = module
    spec.loader.exec_module(module)
    return module


def test_grid_search_selects_best_candidate():
    wfv = load_wfv()
    df = pd.DataFrame({"Close": range(8)})
    grid = {"tp": [1, 2]}

    def bt(_df, tp=1):
        # pnl higher and drawdown lower for tp=2
        return {"pnl": tp * 2.0, "winrate": 0.6, "maxdd": 1.0 / tp}

    res = wfv.walk_forward_grid_search(
        df,
        grid,
        bt,
        train_window=4,
        test_window=2,
        step=2,
        objective_metrics=["pnl", "maxdd"],
    )
    assert res["tp"].iloc[0] == 2


def test_prune_features_no_drop():
    wfv = load_wfv()
    df = pd.DataFrame({"a": [1], "b": [2]})
    imp = {"a": 0.5, "b": 0.3}
    new_df, dropped = wfv.prune_features_by_importance(df, imp, threshold=0.1)
    assert dropped == []
    assert list(new_df.columns) == ["a", "b"]


def test_optuna_fold_overlap_error():
    wfv = load_wfv()
    df = pd.DataFrame({"Close": range(6)}, index=[0, 1, 2, 2, 3, 4])
    space = {"signal": (0.5, 1.0, 0.5)}
    import optuna

    config = types.SimpleNamespace(optuna=optuna)
    sys_modules_backup = sys.modules.get("src.config")
    sys.modules["src.config"] = config
    try:
        with pytest.raises(ValueError):
            wfv.optuna_walk_forward_per_fold(
                df,
                space,
                lambda d, **p: {"pnl": 1.0},
                train_window=3,
                test_window=2,
                step=1,
                n_trials=1,
            )
    finally:
        if sys_modules_backup is not None:
            sys.modules["src.config"] = sys_modules_backup
        else:
            del sys.modules["src.config"]
