import os
import sys
import numpy as np
import pandas as pd
import types
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features
import src.data_loader as dl


def test_select_top_shap_features_basic():
    shap_vals = np.array([[0.2, 0.01, 0.05], [0.3, 0.02, 0.01]], dtype='float32')
    feats = ['f1', 'f2', 'f3']
    selected = features.select_top_shap_features(shap_vals, feats, shap_threshold=0.1)
    assert selected == ['f1', 'f3']


def test_select_top_shap_features_empty_array():
    shap_vals = np.empty((0, 3), dtype='float32')
    feats = ['a', 'b', 'c']
    selected = features.select_top_shap_features(shap_vals, feats)
    assert selected == feats


def test_select_top_shap_features_invalid_feature_names():
    shap_vals = np.array([[0.1]], dtype='float32')
    assert features.select_top_shap_features(shap_vals, None) is None


def test_check_feature_noise_shap_no_noise(caplog):
    shap_vals = np.array([[0.5, 0.4], [0.5, 0.6]], dtype='float32')
    feats = ['x', 'y']
    with caplog.at_level('INFO'):
        features.check_feature_noise_shap(shap_vals, feats, threshold=0.1)
    assert "No features" in ''.join(caplog.messages)


def test_check_feature_noise_shap_detect_noise(caplog):
    shap_vals = np.array([[0.001, 0.9], [0.002, 1.0]], dtype='float32')
    feats = ['n1', 'n2']
    with caplog.at_level('INFO'):
        features.check_feature_noise_shap(shap_vals, feats, threshold=0.01)
    assert "SHAP Noise" in ''.join(caplog.messages)


def test_analyze_feature_importance_shap_no_shap_lib(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(features, 'shap', None, raising=False)
    model = object()
    df = pd.DataFrame({'f1': [1], 'f2': [2]})
    with caplog.at_level('WARNING'):
        features.analyze_feature_importance_shap(model, 'CatBoostClassifier', df, ['f1', 'f2'], tmp_path)
    assert "library not found" in ''.join(caplog.messages)


def test_analyze_feature_importance_shap_invalid_output_dir(monkeypatch, tmp_path, caplog):
    dummy_shap = types.SimpleNamespace(TreeExplainer=lambda m: None)
    monkeypatch.setattr(features, 'shap', dummy_shap, raising=False)
    model = object()
    df = pd.DataFrame({'f': [1]})
    with caplog.at_level('WARNING'):
        features.analyze_feature_importance_shap(model, 'CatBoostClassifier', df, ['f'], tmp_path / 'missing')
    assert "Output directory" in ''.join(caplog.messages)


def test_check_price_jumps_detects():
    df = pd.DataFrame({'Close': [1.0, 1.2, 1.21, 1.0]})
    assert dl.check_price_jumps(df, threshold=0.1) == 2


def test_load_raw_data_m1_returns_dataframe(tmp_path):
    csv = tmp_path / 'd.csv'
    pd.DataFrame({'A': [1]}).to_csv(csv)
    df = dl.load_raw_data_m1(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_raw_data_m15_returns_dataframe(tmp_path):
    csv = tmp_path / 'd.csv'
    pd.DataFrame({'A': [1]}).to_csv(csv)
    df = dl.load_raw_data_m15(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
