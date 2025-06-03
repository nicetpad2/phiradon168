import os
import sys
import pandas as pd
import numpy as np
import types
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features
import src.data_loader as dl


def test_ema_invalid_input_type_raises():
    with pytest.raises(TypeError):
        features.ema([1, 2, 3], 3)


def test_ema_empty_series_returns_empty():
    result = features.ema(pd.Series(dtype='float32'), 3)
    assert result.empty


def test_ema_all_nan_returns_nan_series():
    series = pd.Series([np.nan, np.inf], dtype='float32')
    result = features.ema(series, 3)
    assert result.isna().all()


def test_rsi_insufficient_data_logs_warning(caplog):
    series = pd.Series([1, 2], dtype='float32')
    with caplog.at_level(logging.WARNING):
        res = features.rsi(series, period=5)
    assert res.isna().all()
    assert any('RSI calculation skipped' in msg for msg in caplog.messages)


def test_atr_missing_columns_logs_warning(caplog):
    df = pd.DataFrame({'Open': [1, 2], 'Close': [1, 2]})
    with caplog.at_level(logging.WARNING):
        res = features.atr(df, period=5)
    assert 'ATR_5' in res.columns
    assert any('ATR calculation skipped: Missing columns' in msg for msg in caplog.messages)


def test_atr_insufficient_data_logs_warning(caplog):
    df = pd.DataFrame({'High': [1], 'Low': [1], 'Close': [1]})
    with caplog.at_level(logging.WARNING):
        res = features.atr(df, period=5)
    assert 'ATR_5' in res.columns
    assert any('ATR calculation skipped: Not enough valid data' in msg for msg in caplog.messages)


def test_macd_insufficient_data_logs_warning(caplog):
    series = pd.Series([1, 2], dtype='float32')
    with caplog.at_level(logging.DEBUG):
        line, signal, diff = features.macd(series, window_slow=5, window_fast=2, window_sign=2)
    assert line.isna().all()
    assert any('Input series too short' in msg for msg in caplog.messages)


def test_rolling_zscore_inf_values_returns_zeros():
    series = pd.Series([np.inf, np.inf], dtype='float32')
    result = features.rolling_zscore(series, 3)
    assert (result == 0.0).all()


def test_get_session_tag_custom_map():
    ts = pd.Timestamp('2024-01-01 13:30', tz='UTC')
    custom = {'Test': (13, 14)}
    assert features.get_session_tag(ts, custom) == 'Test'


def test_safe_get_global_existing(monkeypatch):
    monkeypatch.setattr(dl, 'TEST_VAR', 99, raising=False)
    assert dl.safe_get_global('TEST_VAR', 0) == 99


