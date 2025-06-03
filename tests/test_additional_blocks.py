import os
import sys
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features
import src.data_loader as dl


def test_preview_datetime_format_returns_none(capsys):
    df = pd.DataFrame({'Date': ['2024-01-01'], 'Timestamp': ['12:00']})
    result = dl.preview_datetime_format(df, n=1)
    assert result is None


def test_ema_returns_float32_series():
    series = pd.Series([1, 2, 3], dtype='float32')
    result = features.ema(series, 2)
    assert result.dtype == 'float32'
    assert len(result) == len(series)


def test_sma_handles_nan():
    series = pd.Series([1, np.nan, 3], dtype='float32')
    result = features.sma(series, 2)
    assert len(result) == len(series)


def test_rsi_returns_series(monkeypatch):
    monkeypatch.setattr(features, 'ta', None, raising=False)
    series = pd.Series([1, 2, 3], dtype='float32')
    result = features.rsi(series, period=5)
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)


def test_atr_missing_columns():
    df = pd.DataFrame({'High': [], 'Low': [], 'Close': []})
    result = features.atr(df, period=14)
    assert 'ATR_14' in result.columns


def test_macd_returns_three_series(monkeypatch):
    monkeypatch.setattr(features, 'ta', None, raising=False)
    series = pd.Series([1, 2, 3], dtype='float32')
    line, signal, diff = features.macd(series)
    assert len(line) == len(series)
    assert len(signal) == len(series)
    assert len(diff) == len(series)


def test_tag_price_structure_patterns_empty_df():
    df = pd.DataFrame()
    result = features.tag_price_structure_patterns(df)
    assert (result['Pattern_Label'] == 'Normal').all()


def test_get_session_tag_timezone_naive():
    ts = pd.Timestamp('2024-01-01 10:00')
    tag = features.get_session_tag(ts)
    assert isinstance(tag, str)


def test_set_thai_font_no_error():
    assert dl.set_thai_font() in {True, False}


def test_setup_fonts_creates_base(tmp_path):
    dl.setup_fonts(str(tmp_path))
    assert os.path.isdir(tmp_path)


def test_ema_raises_type_error_non_series():
    with pytest.raises(TypeError):
        features.ema([1, 2, 3], 2)


def test_sma_invalid_period_returns_nan_series():
    series = pd.Series([1, 2, 3], dtype='float32')
    result = features.sma(series, -1)
    assert result.isna().all()


def test_rolling_zscore_non_series_raises():
    with pytest.raises(TypeError):
        features.rolling_zscore([1, 2, 3], 3)



def test_tag_price_structure_patterns_missing_columns():
    df = pd.DataFrame({'Gain_Z': [0], 'High': [1], 'Low': [1], 'Close': [1]})
    result = features.tag_price_structure_patterns(df)
    assert (result['Pattern_Label'] == 'Normal').all()


def test_calculate_m15_trend_zone_empty_df_returns_neutral():
    df = pd.DataFrame()
    result = features.calculate_m15_trend_zone(df)
    assert (result['Trend_Zone'] == 'NEUTRAL').all()


def test_get_session_tag_midnight():
    ts = pd.Timestamp('2024-01-01 00:00', tz='UTC')
    tag = features.get_session_tag(ts)
    assert isinstance(tag, str)


def test_simple_converter_none():
    assert dl.simple_converter(None) is None


def test_setup_output_directory_returns_absolute(tmp_path):
    result = dl.setup_output_directory(str(tmp_path), 'out')
    assert os.path.isabs(result)
    assert os.path.isdir(result)


def test_preview_datetime_format_missing_columns():
    df = pd.DataFrame({'Date': ['2024-01-01']})
    assert dl.preview_datetime_format(df) is None


def test_safe_set_datetime_invalid_value_results_nat():
    df = pd.DataFrame(index=[0])
    dl.safe_set_datetime(df, 0, 'Date', 'invalid')
    assert pd.isna(df.loc[0, 'Date'])
