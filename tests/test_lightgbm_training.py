import os
import pandas as pd
from pathlib import Path
import src.training as training
from sklearn.linear_model import LogisticRegression


def test_train_lightgbm_mtf_missing_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'LGBMClassifier', None, raising=False)
    res = training.train_lightgbm_mtf('m1.csv', 'm15.csv', str(tmp_path))
    assert res is None


def test_train_lightgbm_mtf_basic(tmp_path, monkeypatch):
    timestamps = pd.date_range('2024-01-01', periods=12, freq='1min')
    m1 = pd.DataFrame({
        'timestamp': timestamps,
        'Open': [0,1]*6,
        'High': [0,1]*6,
        'Low': [0,1]*6,
        'Close': [0,1]*6,
        'Volume': 1
    })
    m15_ts = pd.date_range('2024-01-01', periods=4, freq='15min')
    m15 = pd.DataFrame({
        'timestamp': m15_ts,
        'Open': [0,1,0,1],
        'High': [0,1,0,1],
        'Low': [0,1,0,1],
        'Close': [0,1,0,1],
        'Volume': 1
    })
    m1_path = tmp_path / 'm1.csv'
    m15_path = tmp_path / 'm15.csv'
    m1.to_csv(m1_path, index=False)
    m15.to_csv(m15_path, index=False)

    monkeypatch.setattr(training, 'LGBMClassifier', LogisticRegression, raising=False)

    def fake_save(model, out_dir, name):
        Path(out_dir).mkdir(exist_ok=True)
        Path(out_dir, f"{name}.joblib").write_text("x")
    monkeypatch.setattr(training, 'save_model', fake_save)

    res = training.train_lightgbm_mtf(str(m1_path), str(m15_path), str(tmp_path))
    assert res is not None
    assert os.path.exists(res['model_path']['model'])
    assert res['metrics']['auc'] >= 0.7
