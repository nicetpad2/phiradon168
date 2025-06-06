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
# metrics containing the substring "dd" should also be minimized by default
MINIMIZE_METRICS = {"maxdd", "max_dd", "drawdown"}


def _is_minimize_metric(name: str) -> bool:
    """Return ``True`` if the metric should be minimized."""
    name = name.lower()
    return "dd" in name or name in MINIMIZE_METRICS


def _dominates(a: MetricDict, b: MetricDict, metrics: Sequence[str]) -> bool:
    """Return True if ``a`` dominates ``b`` for the given metrics."""
    def score(val: float, name: str) -> float:
        return -val if _is_minimize_metric(name) else val

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


# [Patch v5.6.5] Optuna-based WFV hyperparameter search
def optuna_walk_forward(
    df: pd.DataFrame,
    param_space: Dict[str, Tuple[float, float, float]],
    backtest_func: Callable[[pd.DataFrame, ...], MetricDict],
    train_window: int,
    test_window: int,
    step: int,
    n_trials: int = 10,
    direction: str = "maximize",
    objective_metric: str = "r_multiple",
) -> pd.DataFrame:
    """Run Optuna optimization with walk-forward validation.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed price/feature data.
    param_space : dict
        Mapping of param name to ``(low, high, step)``.
    backtest_func : Callable
        Function returning a metrics dict per fold.
    train_window : int
        Number of rows for the training window.
    test_window : int
        Number of rows for the testing window.
    step : int
        Step size between folds.
    n_trials : int, optional
        Number of Optuna trials. Defaults to 10.
    direction : str, optional
        Optimization direction. Defaults to ``"maximize"``.
    objective_metric : str, optional
        Metric name used as the objective. Defaults to ``"r_multiple"``.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing each trial with parameters and score.
    """
    from src.config import optuna as _optuna

    if _optuna is None:  # pragma: no cover - optuna may be missing
        logger.warning("optuna not available; returning empty results")
        return pd.DataFrame()

    def objective(trial: "_optuna.trial.Trial") -> float:
        params = {}
        for name, (low, high, step_size) in param_space.items():
            if isinstance(low, int) and isinstance(high, int) and float(step_size).is_integer():
                params[name] = trial.suggest_int(name, int(low), int(high), step=int(step_size))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), step=step_size)

        results = []
        start = 0
        while start + train_window + test_window <= len(df):
            test = df.iloc[start + train_window : start + train_window + test_window]
            metrics = backtest_func(test, **params)
            results.append(metrics)
            start += step

        df_res = pd.DataFrame(results)
        return float(df_res.get(objective_metric, df_res.get("pnl", 0.0)).mean())

    sampler = _optuna.samplers.RandomSampler(seed=42)
    study = _optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, **t.params, "value": t.value})

    return pd.DataFrame(rows)


# [Patch v5.8.7] Walk-forward optimization with per-fold Optuna tuning
def optuna_walk_forward_per_fold(
    df: pd.DataFrame,
    param_space: Dict[str, Tuple[float, float, float]],
    backtest_func: Callable[[pd.DataFrame, ...], MetricDict],
    train_window: int,
    test_window: int,
    step: int,
    n_trials: int = 10,
    direction: str = "maximize",
    objective_metric: str = "pnl",
) -> pd.DataFrame:
    """Optimize parameters on each training fold and evaluate on the next test fold.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed price or feature data.
    param_space : dict
        Mapping of parameter name to ``(low, high, step)`` ranges.
    backtest_func : Callable
        Backtest function executed with sampled parameters.
    train_window : int
        Number of rows to use for training/optimization.
    test_window : int
        Number of rows used for testing.
    step : int
        Step size between folds.
    n_trials : int, optional
        Number of Optuna trials per fold. Defaults to 10.
    direction : str, optional
        Optimization direction. Defaults to ``"maximize"``.
    objective_metric : str, optional
        Metric name to optimize. Defaults to ``"pnl"``.

    Returns
    -------
    pd.DataFrame
        Metrics per fold combined with the best parameters.
    """
    from src.config import optuna as _optuna

    if _optuna is None:  # pragma: no cover - optuna may be missing
        logger.warning("optuna not available; returning empty results")
        return pd.DataFrame()

    assert df.index.is_monotonic_increasing, "DataFrame index must be sorted by datetime"

    fold = 0
    start = 0
    rows: List[Dict[str, float]] = []
    while start + train_window + test_window <= len(df):
        train_df = df.iloc[start : start + train_window]
        test_df = df.iloc[start + train_window : start + train_window + test_window]

        if not train_df.index[-1] < test_df.index[0]:
            raise ValueError("Train and test sets overlap. Check window parameters.")

        def objective(trial: "_optuna.trial.Trial") -> float:
            params = {}
            for name, (low, high, step_size) in param_space.items():
                if isinstance(low, int) and isinstance(high, int) and float(step_size).is_integer():
                    params[name] = trial.suggest_int(name, int(low), int(high), step=int(step_size))
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high), step=step_size)
            metrics = backtest_func(train_df, **params)
            return float(metrics.get(objective_metric, metrics.get("pnl", 0.0)))

        sampler = _optuna.samplers.RandomSampler(seed=42)
        study = _optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        metrics = backtest_func(test_df, **best_params)
        rows.append({"fold": fold, **best_params, **metrics})

        fold += 1
        start += step

    return pd.DataFrame(rows)
