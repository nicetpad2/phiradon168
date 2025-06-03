# -*- coding: utf-8 -*-
"""[Patch v1.1.0] Training utilities for hyperparameter sweep."""
import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - fallback if catboost missing
    CatBoostClassifier = None


# [Patch v1.1.0] Real training function using CatBoost (or logistic regression fallback)
def real_train_func(output_dir: str, learning_rate: float = 0.01, depth: int = 6) -> dict:
    """Train a simple model and return model path, used features and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df_X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.25, random_state=42)

    if CatBoostClassifier:
        model = CatBoostClassifier(iterations=100, learning_rate=learning_rate, depth=depth, verbose=False)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    model_filename = f"model_lr{learning_rate}_depth{depth}.joblib"
    model_path = os.path.join(output_dir, model_filename)
    dump(model, model_path)

    return {
        "model_path": {"model": model_path},
        "features": feature_names,
        "metrics": {"accuracy": acc, "auc": auc},
    }
