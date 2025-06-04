# -*- coding: utf-8 -*-
"""Walk-forward utilities for hyperparameter tuning."""

from __future__ import annotations

import itertools
import logging
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


MetricDict = Dict[str, float]


logger = logging.getLogger(__name__)

# metrics that should be minimized when evaluating the Pareto front
MINIMIZE_METRICS = {"maxdd", "max_dd", "drawdown"}


def _dominates(a: MetricDict, b: MetricDict, metrics: Sequence[str]) -> bool:
    """Return True if ``a`` dominates ``b`` for the given metrics."""
    def score(val: float, name: str) -> float:
        return -val if name in MINIMIZE_METRICS else val

    ge = all(score(a.get(m, float("-inf")), m) >= score(b.get(m, float("-inf")), m) for m in metrics)
    gt = any(score(a.get(m, float("-inf")), m) > score(b.get(m, float("-inf")), m) for m in metrics)
    return ge and gt


def walk_forward_grid_search(
    df: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    backtest_func: Callable[[pd.DataFrame, ...], MetricDict],
    train_window: int,
    test_window: int,
    step: int,
    objective_metrics: Sequence[str] | None = None,
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
    objective_metrics : Sequence[str], optional
        List of metric names used to select Pareto-optimal parameters.
        Metrics containing ``"dd"`` are minimized, all others are maximized.

    Returns
    -------
    pd.DataFrame
        Summary of each fold with selected parameters and metrics.
    """
    if objective_metrics is None:
        objective_metrics = ["pnl"]

    assert df.index.is_monotonic_increasing, "DataFrame index must be sorted by datetime"

    results: List[Dict[str, float]] = []
    start = 0
    fold = 0
    df = df.copy()
    while start + train_window + test_window <= len(df):
        train = df.iloc[start : start + train_window]
        test = df.iloc[start + train_window : start + train_window + test_window]

        train_results = []
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            metrics = backtest_func(train, **params)
            train_results.append({"params": params, "metrics": metrics})

        pareto_candidates = []
        for cand in train_results:
            if not any(
                _dominates(other["metrics"], cand["metrics"], objective_metrics)
                for other in train_results
                if other is not cand
            ):
                pareto_candidates.append(cand)

        if pareto_candidates:
            best_candidate = max(
                pareto_candidates,
                key=lambda c: c["metrics"].get("pnl", float("-inf")),
            )
        else:
            best_candidate = train_results[0]

        best_params = best_candidate["params"]
        metrics = backtest_func(test, **best_params)

        fold += 1
        logger.info(
            "Fold %d: train_range=%s to %s, test_range=%s to %s, best_params=%s, pnl=%.6f, max_dd=%.6f",
            fold,
            train.index[0],
            train.index[-1],
            test.index[0],
            test.index[-1],
            best_params,
            float(metrics.get("pnl", float("nan"))),
            float(metrics.get("maxdd", float("nan"))),
        )
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
