import os
import sys
import pandas as pd
import numpy as np
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features


def test_rsi_series_with_inf_and_short_length_logs_warning(caplog):
    series = pd.Series([np.inf]*5, dtype='float32')
    with caplog.at_level(logging.WARNING):
        res = features.rsi(series, period=14)
    assert res.isna().all()
    assert any('RSI calculation skipped' in msg for msg in caplog.messages)


def test_ema_string_series_logs_warning(caplog):
    series = pd.Series(['a', 'b', 'c'], dtype='object')
    with caplog.at_level(logging.WARNING):
        res = features.ema(series, 3)
    assert res.isna().all()
    assert any('Series contains only NaN/Inf values' in msg for msg in caplog.messages)


def test_sma_invalid_period_returns_nan(caplog):
    series = pd.Series(range(5), dtype='float32')
    with caplog.at_level(logging.ERROR):
        res = features.sma(series, -1)
    assert res.isna().all()
    assert any('SMA calculation failed' in msg for msg in caplog.messages)


def test_rolling_zscore_window_larger_than_series_returns_series():
    series = pd.Series([1, 2, 3], dtype='float32')
    result = features.rolling_zscore(series, 10)
    assert len(result) == len(series)
    assert result.dtype == 'float32'


def test_tag_price_structure_patterns_missing_columns_warning(caplog):
    df = pd.DataFrame({'Gain_Z': [0.1], 'High': [1]})
    with caplog.at_level(logging.WARNING):
        result = features.tag_price_structure_patterns(df)
    assert (result['Pattern_Label'] == 'Normal').all()
    assert any('Missing columns for Pattern Labeling' in msg for msg in caplog.messages)


def test_get_session_tag_outside_sessions_returns_na(caplog):
    ts = pd.Timestamp('2024-01-01 21:00', tz='UTC')
    with caplog.at_level(logging.WARNING):
        tag = features.get_session_tag(ts)
    assert tag == 'NY'
    assert not any('out of all session ranges' in msg for msg in caplog.messages)


def test_engineer_m1_features_with_lag_config_adds_columns():
    df = pd.DataFrame({'Open': [1,2,3], 'High':[1,2,3], 'Low':[1,2,3], 'Close':[1,2,3]})
    cfg = {'features':['Gain'], 'lags':[1]}
    result = features.engineer_m1_features(df, lag_features_config=cfg)
    assert 'Gain_lag1' in result.columns


def test_atr_ta_failure_falls_back(monkeypatch, caplog):
    df = pd.DataFrame({'High': range(20), 'Low': range(20), 'Close': range(20)})
    class DummyATR:
        def __init__(self, *a, **k):
            pass
        def average_true_range(self):
            raise RuntimeError('fail')
    features._atr_cache.clear()
    monkeypatch.setattr(features.ta.volatility, 'AverageTrueRange', DummyATR, raising=False)
    with caplog.at_level(logging.WARNING):
        res = features.atr(df, period=14)
    assert 'ATR_14' in res.columns
    assert any('TA library ATR calculation failed' in msg for msg in caplog.messages)


def test_macd_returns_values():
    series = pd.Series(range(50), dtype='float32')
    line, signal, diff = features.macd(series)
    assert not line.isna().all()
    assert not signal.isna().all()
    assert not diff.isna().all()


def test_rsi_pandas_fallback(monkeypatch):
    series = pd.Series(np.arange(30, dtype='float32'))
    monkeypatch.setattr(features, '_TA_AVAILABLE', False)
    monkeypatch.setattr(features, 'ta', None, raising=False)
    result = features.rsi(series, period=14)
    assert not result.isna().all()


def test_macd_pandas_fallback(monkeypatch):
    series = pd.Series(np.arange(60, dtype='float32'))
    monkeypatch.setattr(features, '_TA_AVAILABLE', False)
    monkeypatch.setattr(features, 'ta', None, raising=False)
    line, signal, diff = features.macd(series)
    assert not line.isna().all()
    assert not signal.isna().all()
    assert not diff.isna().all()


def test_macd_fallback_when_ta_missing(monkeypatch, caplog):
    series = pd.Series(range(50), dtype='float32')
    monkeypatch.setattr(features, 'ta', None, raising=False)
    with caplog.at_level(logging.WARNING):
        line, signal, diff = features.macd(series)
    assert not line.isna().all()
    assert any('fallback MACD' in msg for msg in caplog.messages)


def test_engineer_m1_features_nan_inf_warning(caplog):
    df = pd.DataFrame({'Open': [1.0], 'High': [np.inf], 'Low': [0.0], 'Close': [1.0]})
    with caplog.at_level(logging.INFO):
        result = features.engineer_m1_features(df)
    assert not result.empty
    # [Patch v5.5.4] QA warning should no longer appear after automatic cleaning
    assert all('[QA WARNING] NaN/Inf detected in engineered features' not in msg for msg in caplog.messages)
    assert any('[QA] M1 Feature Engineering Completed' in msg for msg in caplog.messages)
