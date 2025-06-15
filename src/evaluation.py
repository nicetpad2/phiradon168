import os
import json
from typing import Iterable, Tuple, Dict, Callable, List
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import wasserstein_distance
try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None
from src.config import logger
from src.utils import load_json_with_comments
from src.utils.auto_train_meta_classifiers import (
    auto_train_meta_classifiers,
)


def sortino_ratio(returns: Iterable[float]) -> float:
    """Calculate Sortino ratio of series of returns."""
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        return float('nan')
    downside = r[r < 0]
    downside_std = downside.std(ddof=1)
    mean_ret = r.mean()
    if downside_std == 0:
        return float('inf') if mean_ret > 0 else 0.0
    return mean_ret / downside_std


def calmar_ratio(equity: Iterable[float]) -> float:
    """Calculate Calmar ratio from equity curve."""
    eq = np.asarray(list(equity), dtype=float)
    if eq.size < 2:
        return float('nan')
    returns = np.diff(eq) / eq[:-1]
    max_dd = 0.0
    peak = eq[0]
    for val in eq:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    ann_ret = returns.mean() * 252
    if max_dd == 0:
        return float('inf') if ann_ret > 0 else 0.0
    return ann_ret / max_dd


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray | None:
    """Return SHAP values array if ``shap`` is available."""
    if shap is None:
        logger.warning("shap not installed, skipping SHAP computation")
        return None
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        return np.array(shap_values.values)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("SHAP computation failed: %s", exc)
        return None


def find_best_threshold(
    proba: Iterable[float],
    y_true: Iterable[int],
    step: float = 0.05,
) -> Dict[str, float]:
    """Find threshold that maximizes F1 score and return summary metrics."""
    proba = np.array(list(proba))
    y_true = np.array(list(y_true))
    thresholds = np.arange(0.1, 0.9, step)
    best_t = 0.5
    best_s = 0.0
    best_prec = 0.0
    best_rec = 0.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_s:
            best_s = score
            best_t = t
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)
    return {
        "best_threshold": best_t,
        "best_f1": best_s,
        "precision": best_prec,
        "recall": best_rec,
    }


def evaluate_meta_classifier(
    model_path: str, validation_path: str, features_path: str | None = None
):
    """Evaluate a saved meta-classifier using validation data."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    if not os.path.exists(validation_path):
        logger.error(f"Validation data not found: {validation_path}")
        return None

    if features_path is None:
        features_path = os.path.join(os.path.dirname(model_path), "features_main.json")

    try:
        features = load_json_with_comments(features_path)
        if not isinstance(features, list):
            raise ValueError("Invalid features format")
    except (FileNotFoundError, ValueError) as e:
        logger.error("features_path ไม่ถูกต้อง: %s", e)
        return None
    except Exception as e:
        logger.error(f"Could not load features: {e}")
        return None

    dtype_map = {c: "float32" for c in features}
    try:
        from src.utils.data_utils import safe_read_csv

        df = safe_read_csv(validation_path)
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as e:
        logger.error(f"Failed to load validation data: {e}")
        return None

    if "target" not in df.columns:
        logger.error("Validation data must contain 'target' column")
        return None

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        logger.error(f"Validation data missing features: {missing_cols}")
        return None

    X = df[features]
    y = df["target"]

    try:
        model = load(model_path)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return None

    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return None

    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    logger.info(f"[QA] Meta model evaluation AUC={auc:.4f}, ACC={acc:.4f}")
    return {"accuracy": acc, "auc": auc}


# --- Walk-Forward Overfitting Utilities ---


def walk_forward_yearly_validation(
    df: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame], Dict[str, float]],
    train_years: int = 3,
    test_years: int = 1,
) -> pd.DataFrame:
    """Run walk-forward validation by year windows."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex and not be empty")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    start_year = df.index.min().year
    end_year = df.index.max().year
    results: List[Dict[str, float]] = []
    fold = 1
    for year in range(start_year + train_years - 1, end_year - test_years + 1):
        train_start = year - train_years + 1
        train_end = year
        test_start = year + 1
        test_end = test_start + test_years - 1
        train_df = df[str(train_start) : str(train_end)]
        test_df = df[str(test_start) : str(test_end)]
        if train_df.empty or test_df.empty:
            continue
        train_m = backtest_func(train_df)
        test_m = backtest_func(test_df)
        results.append(
            {
                "fold": fold,
                "train_period": f"{train_start}-{train_end}",
                "test_period": f"{test_start}-{test_end}",
                "train_winrate": float(train_m.get("winrate", float("nan"))),
                "train_pnl": float(train_m.get("pnl", float("nan"))),
                "test_winrate": float(test_m.get("winrate", float("nan"))),
                "test_pnl": float(test_m.get("pnl", float("nan"))),
                "test_maxdd": float(test_m.get("maxdd", float("nan"))),
            }
        )
        fold += 1
    return pd.DataFrame(results)


def detect_overfit_wfv(results: pd.DataFrame, threshold: float = 0.2) -> bool:
    """Return True if training PnL vastly exceeds test PnL."""
    if results.empty:
        raise ValueError("results dataframe is empty")
    train_avg = results["train_pnl"].mean()
    test_avg = results["test_pnl"].mean()
    if train_avg <= 0:
        return False
    drop_ratio = (train_avg - test_avg) / (abs(train_avg) + 1e-9)
    return drop_ratio > threshold and test_avg <= 0


# [Patch v6.1.7] Calculate Wasserstein drift by time period
def calculate_drift_by_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    period: str = "D",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return per-period Wasserstein distances for numeric features."""
    if not isinstance(train_df.index, pd.DatetimeIndex) or not isinstance(
        test_df.index, pd.DatetimeIndex
    ):
        raise ValueError("DataFrames must have DatetimeIndex")
    if threshold is None:
        from src.config import DRIFT_WASSERSTEIN_THRESHOLD as _thr

        threshold = _thr

    records = []
    common = [
        c
        for c in train_df.columns
        if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    for col in common:
        train_grp = train_df[col].groupby(train_df.index.to_period(period)).mean()
        test_grp = test_df[col].groupby(test_df.index.to_period(period)).mean()
        for p in train_grp.index.intersection(test_grp.index):
            w = wasserstein_distance([train_grp[p]], [test_grp[p]])
            records.append(
                {
                    "period": str(p),
                    "feature": col,
                    "wasserstein": float(w),
                    "drift": bool(w > threshold),
                }
            )
    return pd.DataFrame(records)


# [Patch] Daily/weekly drift summary helper
def calculate_drift_summary(
    train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Return combined daily and weekly drift report."""
    daily = calculate_drift_by_period(
        train_df, test_df, period="D", threshold=threshold
    )
    daily["period_type"] = "D"
    weekly = calculate_drift_by_period(
        train_df, test_df, period="W", threshold=threshold
    )
    weekly["period_type"] = "W"
    report = pd.concat([daily, weekly], ignore_index=True)
    if not report.empty and report["drift"].any():
        drift_feats = sorted(report.loc[report["drift"], "feature"].unique())
        logger.warning("Drift detected: %s", drift_feats)
    return report
