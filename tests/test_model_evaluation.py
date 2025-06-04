import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
from src.evaluation import evaluate_meta_classifier


def test_evaluate_meta_classifier_success(tmp_path):
    data = pd.DataFrame({
        'f1': [0, 1, 0, 1],
        'f2': [1, 1, 0, 0],
        'target': [0, 1, 0, 1],
    })
    val_path = tmp_path / 'val.csv'
    data.to_csv(val_path, index=False)

    model = LogisticRegression().fit(data[['f1', 'f2']], data['target'])
    model_path = tmp_path / 'meta_classifier.pkl'
    dump(model, model_path)

    with open(tmp_path / 'features_main.json', 'w', encoding='utf-8') as f:
        json.dump(['f1', 'f2'], f)

    metrics = evaluate_meta_classifier(str(model_path), str(val_path))
    assert metrics is not None
    assert metrics['accuracy'] >= 0.9
    assert metrics['auc'] >= 0.9


def test_evaluate_meta_classifier_missing_feature(tmp_path):
    data = pd.DataFrame({
        'f1': [0, 1],
        'target': [0, 1],
    })
    val_path = tmp_path / 'val.csv'
    data.to_csv(val_path, index=False)

    model = LogisticRegression().fit(data[['f1']], data['target'])
    model_path = tmp_path / 'meta_classifier.pkl'
    dump(model, model_path)

    with open(tmp_path / 'features_main.json', 'w', encoding='utf-8') as f:
        json.dump(['f1', 'f2'], f)

    metrics = evaluate_meta_classifier(str(model_path), str(val_path))
    assert metrics is None
