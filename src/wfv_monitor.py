# -*- coding: utf-8 -*-
"""Walk-forward validation with KPI checks."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable

from src.evaluation import calculate_drift_by_period
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

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


def walk_forward_loop(
    df: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame, pd.DataFrame], MetricDict],
    kpi: Dict[str, float],
    train_window: int,
    test_window: int,
    step: int,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run sliding-window walk-forward validation and log each fold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่จัดเรียงตามเวลา
    backtest_func : Callable
        ฟังก์ชัน backtest ที่รับ ``train_df`` และ ``test_df``
    kpi : dict
        เกณฑ์ KPI ที่ใช้ตรวจสอบเช่น ``profit`` และ ``winrate``
    train_window : int
        ขนาดหน้าต่างข้อมูลสำหรับฝึก
    test_window : int
        ขนาดหน้าต่างข้อมูลสำหรับทดสอบ
    step : int
        จำนวนแถวที่เลื่อนไปในแต่ละรอบ
    output_path : str, optional
        หากระบุจะบันทึกผลแต่ละ fold ลง CSV

    Returns
    -------
    pd.DataFrame
        สรุปผลลัพธ์ของแต่ละ fold
    """

    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted")

    rows = []
    start = 0
    fold = 0
    while start + train_window + test_window <= len(df):
        train_df = df.iloc[start : start + train_window]
        test_df = df.iloc[start + train_window : start + train_window + test_window]
        metrics = backtest_func(train_df, test_df)

        failed = (
            metrics.get("pnl", 0.0) < kpi.get("profit", float("-inf"))
            or metrics.get("winrate", 0.0) < kpi.get("winrate", 0.0)
            or metrics.get("maxdd", 0.0) > kpi.get("maxdd", float("inf"))
            or metrics.get("auc", 0.0) < kpi.get("auc", 0.0)
        )
        row = {"fold": fold, **metrics, "failed": failed}
        rows.append(row)

        if output_path is not None:
            df_out = pd.DataFrame([row])
            df_out.to_csv(output_path, mode="a", header=not Path(output_path).exists(), index=False)

        start += step
        fold += 1

    return pd.DataFrame(rows)


# [Patch v6.1.8] Drift monitoring helper
def monitor_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    period: str = "D",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Calculate drift by period and log warnings if exceeded."""

    res = calculate_drift_by_period(train_df, test_df, period=period, threshold=threshold)
    if not res.empty and res["drift"].any():
        features = sorted(res.loc[res["drift"], "feature"].unique())
        logger.warning("Data drift detected for features: %s", features)
    return res


# [Patch v6.2.1] Daily/weekly drift monitoring summary
def monitor_drift_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Calculate daily and weekly drift summary and log warnings."""

    from src.evaluation import calculate_drift_summary

    res = calculate_drift_summary(train_df, test_df, threshold=threshold)
    if not res.empty and res["drift"].any():
        feats = sorted(res.loc[res["drift"], "feature"].unique())
        logger.warning("Data drift summary detected for features: %s", feats)
    return res


# [Patch v6.2.2] Monitor AUC drop by period
def monitor_auc_drop(
    df: pd.DataFrame,
    threshold: float,
    period: str = "D",
    proba_col: str = "proba",
    target_col: str = "target",
) -> pd.DataFrame:
    """Compute AUC per period and warn when below threshold."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex")
    if proba_col not in df.columns or target_col not in df.columns:
        raise ValueError("missing required columns")

    rows = []
    for p, g in df.groupby(df.index.to_period(period)):
        if g[target_col].nunique() < 2:
            continue
        auc = roc_auc_score(g[target_col], g[proba_col])
        rows.append(
            {"period": str(p), "auc": float(auc), "below_threshold": bool(auc < threshold)}
        )

    res = pd.DataFrame(rows)
    if not res.empty and res["below_threshold"].any():
        periods = res.loc[res["below_threshold"], "period"].tolist()
        logger.warning("AUC drop detected: %s", periods)
    return res
