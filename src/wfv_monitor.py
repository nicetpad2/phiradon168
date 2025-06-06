# -*- coding: utf-8 -*-
"""Walk-forward validation with KPI checks."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

MetricDict = Dict[str, float]


def walk_forward_validate(
    df: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame, pd.DataFrame], MetricDict],
    kpi: Dict[str, float],
    n_splits: int = 5,
    retrain_func: Callable[[int, MetricDict], None] | None = None,
) -> pd.DataFrame:
    """Perform walk-forward validation and trigger retraining on KPI failure.

    Parameters
    ----------
    df : pd.DataFrame
        Time ordered dataframe containing features and target.
    backtest_func : Callable
        Function that accepts ``train_df`` and ``test_df`` and returns metrics
        including ``pnl``, ``winrate``, ``maxdd`` and ``auc``.
    kpi : dict
        Thresholds for metrics: ``profit``, ``winrate``, ``maxdd``, ``auc``.
    n_splits : int, optional
        Number of folds. Defaults to 5.
    retrain_func : callable, optional
        Callback executed when a fold fails KPI. Receives ``fold`` index and
        ``metrics`` dict. Defaults to ``None``.

    Returns
    -------
    pd.DataFrame
        Metrics per fold with a ``failed`` column indicating KPI breaches.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted")

    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        metrics = backtest_func(train_df, test_df)

        fail = (
            metrics.get("pnl", 0.0) < kpi.get("profit", float("-inf"))
            or metrics.get("winrate", 0.0) < kpi.get("winrate", 0.0)
            or metrics.get("maxdd", 0.0) > kpi.get("maxdd", float("inf"))
            or metrics.get("auc", 0.0) < kpi.get("auc", 0.0)
        )
        if fail and retrain_func is not None:
            try:
                retrain_func(fold, metrics)
            except Exception as exc:  # pragma: no cover - retrain may fail silently
                logger.error("Retrain callback failed: %s", exc)
        results.append({"fold": fold, **metrics, "failed": fail})

    return pd.DataFrame(results)
