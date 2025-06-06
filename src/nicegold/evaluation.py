import os
import json
from nicegold.utils import load_json_with_comments
from typing import Iterable, Tuple, Dict
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
from nicegold.config import logger


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
    """Evaluate a saved meta-classifier using validation data.

    Parameters
    ----------
    model_path : str
        Path to the trained model (.pkl or .joblib).
    validation_path : str
        CSV file with feature columns and a 'target' column.
    features_path : str, optional
        JSON file listing feature names. Defaults to 'features_main.json' next to the model file.

    Returns
    -------
    dict | None
        Dictionary with 'accuracy' and 'auc' if evaluation succeeds, otherwise None.
    """
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
        # [Patch] clearer log when features_path is invalid
        logger.error("features_path ไม่ถูกต้อง: %s", e)
        return None
    except Exception as e:
        logger.error(f"Could not load features: {e}")
        return None

    dtype_map = {c: "float32" for c in features}
    try:
        # [Patch] specify dtype to avoid memory spike
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
