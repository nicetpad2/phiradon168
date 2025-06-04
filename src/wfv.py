# -*- coding: utf-8 -*-
"""Walk-forward utilities for hyperparameter tuning."""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd


MetricDict = Dict[str, float]


def walk_forward_grid_search(
    df: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    backtest_func: Callable[[pd.DataFrame, ...], MetricDict],
    train_window: int,
    test_window: int,
    step: int,
) -> pd.DataFrame:
    """Run rolling-window walk-forward validation with grid search.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame of price/feature data.
    param_grid : dict
        Mapping of parameter name to iterable of values.
    backtest_func : Callable
        Function executed for each parameter set. It should return a
        dictionary with at least ``pnl``, ``winrate``, and ``maxdd`` keys.
    train_window : int
        Number of rows used for training.
    test_window : int
        Number of rows used for testing.
    step : int
        Step size between folds.

    Returns
    -------
    pd.DataFrame
        Summary of each fold with selected parameters and metrics.
    """
    results: List[Dict[str, float]] = []
    start = 0
    df = df.reset_index(drop=True)
    while start + train_window + test_window <= len(df):
        train = df.iloc[start : start + train_window]
        test = df.iloc[start + train_window : start + train_window + test_window]

        best_params: Dict[str, float] | None = None
        best_pnl = float("-inf")
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            metrics = backtest_func(train, **params)
            pnl = float(metrics.get("pnl", float("-inf")))
            if pnl > best_pnl:
                best_pnl = pnl
                best_params = params
        if best_params is None:
            best_params = {k: list(v)[0] for k, v in param_grid.items()}
            metrics = {"pnl": float("nan"), "winrate": float("nan"), "maxdd": float("nan")}
        else:
            metrics = backtest_func(test, **best_params)
        results.append(
            {
                "start": start,
                **best_params,
                "pnl": float(metrics.get("pnl", float("nan"))),
                "winrate": float(metrics.get("winrate", float("nan"))),
                "maxdd": float(metrics.get("maxdd", float("nan"))),
            }
        )
        start += step
    return pd.DataFrame(results)


def prune_features_by_importance(
    df: pd.DataFrame,
    importances: Dict[str, float],
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop features with importance below ``threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature columns.
    importances : dict
        Mapping of feature name to importance value.
    threshold : float
        Minimum importance to keep a feature.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        New DataFrame and list of dropped features.
    """
    drop_cols = [f for f, v in importances.items() if v < threshold and f in df.columns]
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols
