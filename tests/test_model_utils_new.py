import pandas as pd
import numpy as np
import logging
from pathlib import Path
import urllib.request
import pytest
from src.utils.model_utils import (
    save_model,
    load_model,
    evaluate_model,
    predict,
    download_model_if_missing,
    download_feature_list_if_missing,
    validate_file,
)


class SimpleModel:
    """Minimal model implementing predict_proba."""

    def fit(self, X, y):
        self.p = float(np.mean(y))

    def predict_proba(self, X):
        proba = np.tile([1 - self.p, self.p], (len(X), 1))
        return proba


def test_save_load_and_predict(tmp_path, monkeypatch):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = [0, 1]
    model = SimpleModel()
    model.fit(X, y)
    path = tmp_path / 'm.pkl'
    save_model(model, str(path))
    loaded = load_model(str(path))
    monkeypatch.setattr('src.utils.model_utils.accuracy_score', lambda a, b: 1.0)
    monkeypatch.setattr('src.utils.model_utils.roc_auc_score', lambda a, b: 0.5)
    prob = predict(loaded, X.iloc[[0]])
    assert 0.0 <= prob <= 1.0
    acc, auc = evaluate_model(loaded, X, y)
    assert acc >= 0


def test_predict_with_class_idx(tmp_path, monkeypatch):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = [0, 1]
    model = SimpleModel()
    model.fit(X, y)
    path = tmp_path / 'm.pkl'
    save_model(model, str(path))
    loaded = load_model(str(path))
    monkeypatch.setattr('src.utils.model_utils.accuracy_score', lambda a, b: 1.0)
    monkeypatch.setattr('src.utils.model_utils.roc_auc_score', lambda a, b: 0.5)
    p0 = predict(loaded, X.iloc[[0]], class_idx=0)
    p1 = predict(loaded, X.iloc[[0]], class_idx=1)
    assert abs(p0 + p1 - 1.0) < 1e-6


def test_evaluate_model_without_proba(caplog):
    class Dummy:
        def predict(self, X):
            return [0] * len(X)

    df = pd.DataFrame({'a': [0], 'b': [1]})
    dummy = Dummy()
    with caplog.at_level(logging.ERROR):
        result = evaluate_model(dummy, df, [0])
    assert result is None
    assert any('support predict_proba' in m for m in caplog.messages)


def test_load_model_missing_file(tmp_path, caplog):
    missing = tmp_path / 'none.pkl'
    with caplog.at_level(logging.ERROR):
        with pytest.raises(FileNotFoundError):
            load_model(str(missing))
    assert any('Model file not found' in m for m in caplog.messages)


def test_download_model_if_missing(monkeypatch, tmp_path):
    path = tmp_path / 'model.pkl'
    monkeypatch.setenv('MODEL_URL', 'http://example.com/model.pkl')
    calls = []

    def fake_download(url, dest):
        calls.append((url, dest))
        Path(dest).write_text('model')

    monkeypatch.setattr(urllib.request, 'urlretrieve', fake_download)
    assert download_model_if_missing(str(path), 'MODEL_URL') is True
    assert path.exists()
    assert calls


def test_download_model_if_missing_exists(monkeypatch, tmp_path):
    path = tmp_path / 'model.pkl'
    path.write_text('x')
    called = False

    def fake(url, dest):
        nonlocal called
        called = True

    monkeypatch.setattr(urllib.request, 'urlretrieve', fake)
    monkeypatch.setenv('MODEL_URL', 'http://example.com/model.pkl')
    assert download_model_if_missing(str(path), 'MODEL_URL') is True
    assert not called


def test_download_model_env_missing(monkeypatch, tmp_path, caplog):
    path = tmp_path / 'model.pkl'
    monkeypatch.delenv('MODEL_URL', raising=False)
    with caplog.at_level(logging.WARNING):
        assert not download_model_if_missing(str(path), 'MODEL_URL')
    assert any('No URL specified' in m for m in caplog.messages)


def test_download_feature_list_error(monkeypatch, tmp_path, caplog):
    path = tmp_path / 'features.json'
    monkeypatch.setenv('FEAT_URL', 'http://example.com/f.json')

    def fail(url, dest):
        raise RuntimeError('fail')

    monkeypatch.setattr(urllib.request, 'urlretrieve', fail)
    with caplog.at_level(logging.WARNING):
        assert not download_feature_list_if_missing(str(path), 'FEAT_URL')
    assert any('Failed to download feature list' in m for m in caplog.messages)


def test_download_feature_list_env_missing(monkeypatch, tmp_path, caplog):
    path = tmp_path / 'features.json'
    monkeypatch.delenv('FEAT_URL', raising=False)
    with caplog.at_level(logging.WARNING):
        assert not download_feature_list_if_missing(str(path), 'FEAT_URL')
    assert any('No URL specified' in m for m in caplog.messages)


def test_download_feature_list_if_missing_success(monkeypatch, tmp_path):
    path = tmp_path / 'features.json'
    monkeypatch.setenv('FEAT_URL', 'http://example.com/f.json')

    def fake(url, dest):
        Path(dest).write_text('data')

    monkeypatch.setattr(urllib.request, 'urlretrieve', fake)
    assert download_feature_list_if_missing(str(path), 'FEAT_URL') is True
    assert path.exists()


def test_download_feature_list_if_missing_exists(monkeypatch, tmp_path):
    path = tmp_path / 'features.json'
    path.write_text('x')
    called = False

    def fake(url, dest):
        nonlocal called
        called = True

    monkeypatch.setattr(urllib.request, 'urlretrieve', fake)
    monkeypatch.setenv('FEAT_URL', 'http://example.com/f.json')
    assert download_feature_list_if_missing(str(path), 'FEAT_URL') is True
    assert not called


def test_predict_no_proba(caplog):
    class Dummy:
        def predict(self, X):
            return [0]

    with caplog.at_level(logging.ERROR):
        result = predict(Dummy(), pd.DataFrame({'a': [1]}))
    assert result is None
    assert any('support predict_proba' in m for m in caplog.messages)


def test_validate_file(tmp_path):
    f = tmp_path / 'f.txt'
    assert not validate_file(str(f))
    f.write_text('')
    assert not validate_file(str(f))
    f.write_text('hi')
    assert validate_file(str(f))
