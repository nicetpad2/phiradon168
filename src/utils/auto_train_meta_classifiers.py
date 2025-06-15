"""
[Patch v6.3.0] Stub for auto-training meta-classifiers.

This module currently provides a placeholder function to be expanded in future
patches.
"""

from typing import Any
import os
import glob
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import logger
from src.utils import load_json_with_comments


def auto_train_meta_classifiers(
    config: Any,
    training_data: Any | None = None,
    models_dir: str | None = None,
    features_dir: str | None = None,
) -> dict | None:
    """[Patch v6.5.5] Auto train meta-classifiers if trade log exists."""
    if training_data is None:
        pattern = os.path.join(config.OUTPUT_DIR, "trade_log_v32_walkforward*.csv.gz")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(config.OUTPUT_DIR, "trade_log_v32_walkforward*.csv")
            matches = glob.glob(pattern)
        if not matches:
            logger.error(
                "[Patch v6.4.2] Walk-forward trade log not found; skipping training."
            )
            return None
        trade_log_path = matches[0]
        logger.info("[Patch v6.4.2] Loading trade log from %s", trade_log_path)
        compression = "gzip" if trade_log_path.endswith(".gz") else None
        try:
            from src.utils.data_utils import safe_read_csv

            training_data = safe_read_csv(trade_log_path)
        except Exception as e:  # pragma: no cover - trivial log path
            logger.error("[Patch v6.4.2] Failed to load %s: %s", trade_log_path, e)
            return None

    if not isinstance(training_data, pd.DataFrame):
        logger.error("[Patch v6.5.5] Training data must be a DataFrame")
        return None

    # ตรวจสอบคอลัมน์กำไรว่าไม่เป็นศูนย์ทั้งหมด ก่อนเริ่มฝึก
    profit_col = next(
        (
            c
            for c in ("profit", "pnl_usd_net", "PnL", "pnl")
            if c in training_data.columns
        ),
        None,
    )
    if profit_col and (training_data[profit_col] == 0).all():
        logger.error(
            "[Patch v6.6.12] All profit values are 0 – skipping meta-classifier training"
        )
        return None

    if features_dir is None:
        features_dir = getattr(config, "OUTPUT_DIR", ".")
    features_path = os.path.join(features_dir, "features_main.json")
    try:
        features = load_json_with_comments(features_path)
        if not isinstance(features, list):
            raise ValueError("Invalid format")
    except Exception as e:
        logger.error("[Patch v6.5.5] Failed to load features file: %s", e)
        return None

    # Ensure 'target' column exists before proceeding
    if "target" not in training_data.columns:
        profit_col = next(
            (
                c
                for c in ("profit", "pnl_usd_net", "PnL", "pnl")
                if c in training_data.columns
            ),
            None,
        )
        if profit_col:
            logger.info(
                "[Patch v6.6.11] Auto-generating 'target' from '%s' : profit > 0 -> 1, else 0",
                profit_col,
            )
            training_data = training_data.copy()
            training_data["target"] = (training_data[profit_col] > 0).astype(int)
        else:
            logger.warning(
                "[Patch v6.5.10] 'target' column missing and no profit-like column found – skip meta-classifier training"
            )
            return {}

    missing = [f for f in features if f not in training_data.columns]
    if missing:
        logger.warning("[Patch v6.6.7] Training data missing features: %s", missing)
    features = [f for f in features if f in training_data.columns]
    if len(features) == 0:
        if profit_col and profit_col in training_data.columns:
            logger.warning(
                "[Patch v6.6.13] No feature columns found; using '%s' as fallback feature",
                profit_col,
            )
            features = [profit_col]
        else:
            logger.error(
                "[Patch v6.6.7] No available features in training data, skipping meta-classifier"
            )
            return {}

    if len(training_data) < 5:
        logger.error(
            "[Patch v6.5.5] Insufficient rows for training: %d", len(training_data)
        )
        return None

    X = training_data[features]
    y = training_data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y, proba)

    if models_dir is None:
        models_dir = getattr(config, "OUTPUT_DIR", ".")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "meta_classifier.joblib")
    dump(model, model_path)
    logger.info("[Patch v6.5.5] Meta-classifier trained: %s", model_path)
    logger.info(
        "[Patch v6.6.6] Meta-classifier metrics - accuracy: %.4f, auc: %.4f", acc, auc
    )
    return {"model_path": model_path, "metrics": {"accuracy": acc, "auc": auc}}
