import os
import sys
import pandas as pd
import numpy as np
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features


def test_ema_all_nan_logs_warning(caplog):
    series = pd.Series([np.nan, np.inf], dtype='float32')
    with caplog.at_level(logging.WARNING):
        res = features.ema(series, 3)
    assert res.isna().all()
    assert any('NaN/Inf values' in msg for msg in caplog.messages)


def test_rsi_ta_not_loaded_error(monkeypatch, caplog):
    series = pd.Series([1, 2, 3], dtype='float32')
    monkeypatch.setattr(features, 'ta', None, raising=False)
    with caplog.at_level(logging.WARNING):
        res = features.rsi(series, period=14)
    assert res.isna().all()
    assert any("fallback RSI" in msg for msg in caplog.messages)


def test_get_session_tag_nat():
    assert features.get_session_tag(pd.NaT) == 'N/A'


def test_get_session_tag_missing_global(monkeypatch, caplog):
    monkeypatch.delattr(features, 'SESSION_TIMES_UTC', raising=False)
    from src import utils
    monkeypatch.delattr(utils.sessions, 'SESSION_TIMES_UTC', raising=False)
    ts = pd.Timestamp('2024-01-01 05:00', tz='UTC')
    with caplog.at_level(logging.WARNING):
        tag = utils.sessions.get_session_tag(ts)
    assert tag == 'Asia'
    assert any('Global SESSION_TIMES_UTC not found' in msg for msg in caplog.messages)


def test_calculate_m15_trend_zone_missing_close():
    df = pd.DataFrame({'Open': [1]})
    result = features.calculate_m15_trend_zone(df)
    assert (result['Trend_Zone'] == 'NEUTRAL').all()


def test_engineer_m1_features_empty_df_warning(caplog):
    df = pd.DataFrame()
    with caplog.at_level(logging.WARNING):
        result = features.engineer_m1_features(df)
    assert result.empty
    assert any('ข้ามการสร้าง Features M1' in msg for msg in caplog.messages)


def test_engineer_m1_features_missing_price_cols_warning(caplog):
    df = pd.DataFrame({'A': [1]})
    with caplog.at_level(logging.WARNING):
        result = features.engineer_m1_features(df)
    assert 'Candle_Body' in result.columns
    assert any('ขาดคอลัมน์ราคา M1' in msg for msg in caplog.messages)


def test_clean_m1_data_empty_df_warning(caplog):
    df = pd.DataFrame()
    with caplog.at_level(logging.WARNING):
        cleaned, feats = features.clean_m1_data(df)
    assert cleaned.empty and feats == []
    assert any('ข้ามการทำความสะอาดข้อมูล M1' in msg for msg in caplog.messages)


def test_clean_m1_data_inf_values_warning(caplog):
    df = pd.DataFrame({'A': [np.inf]})
    with caplog.at_level(logging.WARNING):
        cleaned, _ = features.clean_m1_data(df)
    assert not np.isinf(cleaned['A']).any()
    assert any('Inf Check' in msg for msg in caplog.messages)


def test_macd_ta_not_loaded_error(monkeypatch, caplog):
    series = pd.Series(range(30), dtype='float32')
    monkeypatch.setattr(features, 'ta', None, raising=False)
    with caplog.at_level(logging.WARNING):
        line, signal, diff = features.macd(series)
    assert not line.isna().all() and not signal.isna().all() and not diff.isna().all()
    assert any("fallback MACD" in msg for msg in caplog.messages)
