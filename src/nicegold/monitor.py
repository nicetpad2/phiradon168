import os
import csv
from datetime import datetime, timezone
from typing import Iterable, List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from nicegold.config import logger, LOG_DIR


def _write_row(path: str, row: List[str]):
    header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(["timestamp", "label", "auc", "accuracy"])
        writer.writerow(row)


def log_performance_metrics(
    y_true: Iterable[int],
    proba: Iterable[float],
    label: str = "daily",
    summary_path: str = os.path.join(LOG_DIR, "performance_metrics.csv"),
) -> Dict[str, float]:
    y_true = np.array(list(y_true))
    proba = np.array(list(proba))
    acc = accuracy_score(y_true, proba >= 0.5)
    auc = roc_auc_score(y_true, proba)
    logger.info(f"[Monitor] {label} AUC={auc:.4f}, ACC={acc:.4f}")
    _write_row(
        summary_path,
        [datetime.now(timezone.utc).isoformat(), label, f"{auc:.4f}", f"{acc:.4f}"]
    )
    return {"auc": auc, "accuracy": acc}


def monitor_auc_from_csv(path: str, label: str = "daily", summary_path: str = os.path.join(LOG_DIR, "performance_metrics.csv")) -> Dict[str, float] | None:
    if not os.path.exists(path):
        logger.error("metrics file not found: %s", path)
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if "proba" not in data.dtype.names or "target" not in data.dtype.names:
        logger.error("metrics file missing required columns")
        return None
    return log_performance_metrics(data["target"], data["proba"], label, summary_path)

