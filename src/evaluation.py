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
from src.config import logger
from src.utils import load_json_with_comments


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


def evaluate_meta_classifier(model_path: str, validation_path: str, features_path: str | None = None):
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
        df = pd.read_csv(validation_path, dtype=dtype_map, low_memory=False)
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
        train_df = df[str(train_start): str(train_end)]
        test_df = df[str(test_start): str(test_end)]
        if train_df.empty or test_df.empty:
            continue
        train_m = backtest_func(train_df)
        test_m = backtest_func(test_df)
        results.append({
            "fold": fold,
            "train_period": f"{train_start}-{train_end}",
            "test_period": f"{test_start}-{test_end}",
            "train_winrate": float(train_m.get("winrate", float("nan"))),
            "train_pnl": float(train_m.get("pnl", float("nan"))),
            "test_winrate": float(test_m.get("winrate", float("nan"))),
            "test_pnl": float(test_m.get("pnl", float("nan"))),
            "test_maxdd": float(test_m.get("maxdd", float("nan"))),
        })
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
