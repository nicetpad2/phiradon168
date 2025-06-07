import os
import pandas as pd
from pathlib import Path
import src.training as training
from sklearn.linear_model import LogisticRegression


def test_train_lightgbm_mtf_feature_output(tmp_path, monkeypatch):
    timestamps = pd.date_range('2024-01-01', periods=10, freq='1min')
    m1 = pd.DataFrame({
        'timestamp': timestamps,
        'Open': [0, 1]*5,
        'High': [0, 1]*5,
        'Low': [0, 1]*5,
        'Close': [0, 1]*5,
        'Volume': 1
    })
    m15_ts = pd.date_range('2024-01-01', periods=3, freq='15min')
    m15 = pd.DataFrame({
        'timestamp': m15_ts,
        'Open': [0, 1, 0],
        'High': [0, 1, 0],
        'Low': [0, 1, 0],
        'Close': [0, 1, 0],
        'Volume': 1
    })
    m1_path = tmp_path / 'm1.csv'
    m15_path = tmp_path / 'm15.csv'
    m1.to_csv(m1_path, index=False)
    m15.to_csv(m15_path, index=False)

    monkeypatch.setattr(training, '_time_series_cv_auc', lambda *a, **k: 0.9)
    monkeypatch.setattr(training, 'LGBMClassifier', LogisticRegression, raising=False)

    def fake_save(model, out_dir, name):
        Path(out_dir).mkdir(exist_ok=True)
        Path(out_dir, f"{name}.joblib").write_text('x')
    monkeypatch.setattr(training, 'save_model', fake_save)

    res = training.train_lightgbm_mtf(str(m1_path), str(m15_path), str(tmp_path))
    expected = ['Open', 'High', 'Low', 'Close', 'Volume', 'M15_Close', 'Trend_Up']
    assert res['features'] == expected
    assert os.path.exists(res['model_path']['model'])
    assert res['metrics']['auc'] == 0.9
