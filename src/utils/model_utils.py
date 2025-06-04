import os
import logging
import urllib.request
import json
from typing import Iterable, Tuple

import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score, roc_auc_score

# [Patch v5.5.0] Utility to download model files if missing

def download_model_if_missing(model_path, url_env):
    """Download model from URL specified in environment variable if missing."""
    if os.path.exists(model_path):
        return
    url = os.getenv(url_env)
    if not url:
        return
    try:
        logging.info(f"(Info) Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        logging.info("(Success) Model downloaded.")
    except Exception as e:
        logging.error(f"(Error) Failed to download model from {url}: {e}")

# [Patch v5.6.0] Utility to download feature list files if missing
def download_feature_list_if_missing(features_path, url_env):
    """Download feature list from URL specified in environment variable if missing."""
    if os.path.exists(features_path):
        return
    url = os.getenv(url_env)
    if not url:
        return
    try:
        logging.info(f"(Info) Downloading feature list from {url}...")
        urllib.request.urlretrieve(url, features_path)
        logging.info("(Success) Feature list downloaded.")
    except Exception as e:
        logging.error(f"(Error) Failed to download feature list from {url}: {e}")


def save_model(model, path: str) -> None:
    """บันทึกโมเดลไปยังไฟล์"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def load_model(path: str):
    """โหลดโมเดลจากไฟล์"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load(path)


def evaluate_model(model, X: pd.DataFrame, y: Iterable[int]) -> Tuple[float, float]:
    """คำนวณ Accuracy และ AUC"""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan")
    return acc, auc


def predict(model, X: pd.DataFrame) -> float:
    """คืนค่า probability ของ class 1"""
    return float(model.predict_proba(X)[0, 1])
