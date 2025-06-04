import os
import json
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import logger


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
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        if not isinstance(features, list):
            raise ValueError("Invalid features format")
    except Exception as e:
        logger.error(f"Could not load features: {e}")
        return None

    try:
        df = pd.read_csv(validation_path)
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
