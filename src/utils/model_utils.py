import os
import logging
import urllib.request
import json
from typing import Iterable, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score, roc_auc_score

# [Patch v5.6.1] Utility to download model files if missing

def download_model_if_missing(model_path: str, url_env: str) -> bool:
    """Download model from URL specified in environment variable if missing."""
    if os.path.exists(model_path):
        return True
    url = os.getenv(url_env)
    model_name = os.path.basename(model_path)
    if not url:
        msg = f"No URL specified for {model_name}, skipping download"
        logger.warning(msg)
        logging.getLogger().warning(msg)
        return False
    try:
        msg = f"(Info) Downloading model from {url}..."
        logger.info(msg)
        logging.getLogger().warning(msg)
        urllib.request.urlretrieve(url, model_path)
        msg2 = "(Success) Model downloaded."
        logger.info(msg2)
        logging.getLogger().warning(msg2)
        return True
    except Exception as e:  # pragma: no cover - network errors vary
        msg = f"Failed to download model from {url}: {e}"
        logger.warning(msg)
        logging.getLogger().warning(msg)
        return False

# [Patch v5.6.1] Utility to download feature list files if missing
def download_feature_list_if_missing(features_path: str, url_env: str) -> bool:
    """Download feature list from URL specified in environment variable if missing."""
    if os.path.exists(features_path):
        return True
    url = os.getenv(url_env)
    file_name = os.path.basename(features_path)
    if not url:
        msg = f"No URL specified for {file_name}, skipping download"
        logger.warning(msg)
        logging.getLogger().warning(msg)
        return False
    try:
        msg = f"(Info) Downloading feature list from {url}..."
        logger.info(msg)
        logging.getLogger().warning(msg)
        urllib.request.urlretrieve(url, features_path)
        msg2 = "(Success) Feature list downloaded."
        logger.info(msg2)
        logging.getLogger().warning(msg2)
        return True
    except Exception as e:  # pragma: no cover - network errors vary
        msg = f"Failed to download feature list from {url}: {e}"
        logger.warning(msg)
        logging.getLogger().warning(msg)
        return False


def save_model(model: Any, path: str) -> None:
    """Save model object to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def load_model(path: str) -> Any:
    """Load model from disk, logging errors for missing or invalid files."""
    try:
        return load(path)
    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        logging.getLogger().error(f"Model file not found: {path}")
        raise
    except Exception as e:  # pragma: no cover - invalid format
        logger.error(f"Failed to load model from {path}: {e}")
        logging.getLogger().error(f"Failed to load model from {path}: {e}")
        raise


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: Iterable[int],
    class_idx: int = 1,
) -> Optional[Tuple[float, float]]:
    """Return accuracy and AUC for the given model."""
    if not hasattr(model, "predict_proba"):
        logger.error("Model does not support predict_proba")
        logging.getLogger().error("Model does not support predict_proba")
        return None
    proba = model.predict_proba(X)
    if proba.ndim == 2:
        proba = proba[:, class_idx]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return acc, auc


def predict(model: Any, X: pd.DataFrame, class_idx: int = 1) -> Optional[float]:
    """Return probability of the specified class."""
    if not hasattr(model, "predict_proba"):
        logger.error("Model does not support predict_proba")
        logging.getLogger().error("Model does not support predict_proba")
        return None
    proba = model.predict_proba(X)
    return float(proba[0, class_idx])


# [Patch v5.7.3] Utility to validate existence and size of a file
def validate_file(path: str) -> bool:
    """Return True if file exists and is non-empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0
