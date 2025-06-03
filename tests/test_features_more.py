import os
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features


def test_ema_sma_and_zscore():
    series = pd.Series([1, 2, 3, 4, 5], dtype='float32')
    ema_result = features.ema(series, 3)
    sma_result = features.sma(series, 3)
    zscore_result = features.rolling_zscore(series, 3)

    expected_ema = series.ewm(span=3, adjust=False, min_periods=3).mean().astype('float32')
    expected_sma = series.rolling(window=3, min_periods=3).mean().astype('float32')
    rm = series.rolling(window=3, min_periods=2).mean()
    rs = series.rolling(window=3, min_periods=2).std()
    expected_z = ((series - rm) / rs.replace(0, np.nan)).fillna(0.0).astype('float32')

    pd.testing.assert_series_equal(ema_result, expected_ema)
    pd.testing.assert_series_equal(sma_result, expected_sma)
    pd.testing.assert_series_equal(zscore_result.fillna(0.0), expected_z)


def test_macd_with_dummy_ta(monkeypatch):
    class DummyMACD:
        def __init__(self, close, window_slow=26, window_fast=12, window_sign=9, fillna=False):
            self.close = pd.Series(close)
        def macd(self):
            return self.close
        def macd_signal(self):
            return self.close * 0.5
        def macd_diff(self):
            return self.close * 0.1

    monkeypatch.setattr(features.ta.trend, 'MACD', DummyMACD)
    series = pd.Series(np.arange(10), dtype='float32')
    line, signal, diff = features.macd(series, window_slow=5, window_fast=2, window_sign=2)
    assert line.iloc[-1] == series.iloc[-1]
    assert signal.iloc[-1] == series.iloc[-1] * 0.5
    assert diff.iloc[-1] == series.iloc[-1] * 0.1


def test_calculate_m15_trend_zone(monkeypatch):
    idx = pd.date_range('2024-01-01', periods=4, freq='15min')
    df = pd.DataFrame({'Close': [1, 2, 3, 4]}, index=idx)

    def fake_ema(series, period):
        if period == features.M15_TREND_EMA_FAST:
            return pd.Series([3, 3, 2, 2], index=series.index, dtype='float32')
        return pd.Series([2, 2, 3, 3], index=series.index, dtype='float32')

    def fake_rsi(series, period):
        return pd.Series([55, 55, 45, 45], index=series.index, dtype='float32')

    monkeypatch.setattr(features, 'ema', fake_ema)
    monkeypatch.setattr(features, 'rsi', fake_rsi)

    result = features.calculate_m15_trend_zone(df)
    assert list(result['Trend_Zone']) == ['UP', 'UP', 'DOWN', 'DOWN']


def test_clean_m1_data_basic():
    df = pd.DataFrame(
        {
            'Candle_Body': [1.0, np.inf],
            'Pattern_Label': ['Normal', None],
            'session': ['Asia', None],
        }
    )
    cleaned, feats = features.clean_m1_data(df)
    assert cleaned['Candle_Body'].dtype == 'float32'
    assert not np.isinf(cleaned['Candle_Body']).any()
    assert 'Pattern_Label' in feats and 'session' in feats
    assert cleaned['Pattern_Label'].dtype.name == 'category'


def test_calculate_m1_entry_signals():
    df = pd.DataFrame(
        {
            'Gain_Z': [0.4, -0.6],
            'Pattern_Label': ['Breakout', 'Reversal'],
            'RSI': [60, 40],
            'Volatility_Index': [1.0, 1.0],
        }
    )
    config = {
        'gain_z_thresh': 0.3,
        'rsi_thresh_buy': 50,
        'rsi_thresh_sell': 50,
        'volatility_max': 4.0,
        'min_signal_score': 1.0,
    }
    result = features.calculate_m1_entry_signals(df, config)
    assert result['Entry_Long'].iloc[0] == 1
    assert result['Entry_Short'].iloc[1] == 1


def test_engineer_m1_features_full(monkeypatch):
    idx = pd.date_range('2024-01-01', periods=6, freq='1min', tz='UTC')
    df = pd.DataFrame(
        {
            'Open': np.arange(1, 7, dtype='float32'),
            'High': np.arange(1.1, 7.1, 1.0, dtype='float32'),
            'Low': np.arange(0.9, 6.9, 1.0, dtype='float32'),
            'Close': np.arange(1, 7, dtype='float32'),
        },
        index=idx,
    )

    monkeypatch.setattr(features, 'macd', lambda s: (s, s, s))
    monkeypatch.setattr(features, 'atr', lambda df, period=14: df.assign(ATR_14=0.1, ATR_14_Shifted=0.1))
    monkeypatch.setattr(features, 'ema', lambda series, period: series.astype('float32'))
    monkeypatch.setattr(features, 'sma', lambda series, period: series.rolling(period, min_periods=1).mean())
    monkeypatch.setattr(features, 'rsi', lambda series, period=14: pd.Series(np.linspace(40, 60, len(series)), index=series.index))

    class DummyScaler:
        def fit_transform(self, X):
            return X

    class DummyKMeans:
        def __init__(self, n_clusters=3, random_state=42, n_init=10):
            pass

        def fit_predict(self, X):
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(features, 'StandardScaler', DummyScaler)
    monkeypatch.setattr(features, 'KMeans', DummyKMeans)

    result = features.engineer_m1_features(df)
    assert 'cluster' in result.columns
    assert 'session' in result.columns
    assert result['cluster'].dtype.name.startswith('int')
    assert result['session'].dtype.name == 'category'
