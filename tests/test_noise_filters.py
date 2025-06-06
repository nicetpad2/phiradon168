import pandas as pd
import numpy as np
import src.features as features


def test_volatility_filter_basic():
    df = pd.DataFrame({
        'High': [1, 2, 3, 4],
        'Low': [0, 1, 2, 3],
        'Close': [0.5, 1.5, 2.5, 3.5],
        'ATR_14': [0.1, 0.2, 0.3, 0.4],
    })
    res = features.volatility_filter(df, period=14, window=2)
    assert len(res) == len(df)
    assert res.iloc[-1]


def test_median_filter_basic():
    series = pd.Series([1, 100, 3], dtype='float32')
    res = features.median_filter(series, window=3)
    expected = series.rolling(3, min_periods=1).median()
    pd.testing.assert_series_equal(res, expected)


def test_bar_range_filter_basic():
    df = pd.DataFrame({'High': [3, 2], 'Low': [1, 1]})
    res = features.bar_range_filter(df, threshold=1.5)
    assert list(res) == [True, False]


def test_volume_filter_basic():
    df = pd.DataFrame({'Volume': [1, 2, 1, 2]})
    res = features.volume_filter(df, window=2, factor=0.5)
    expected_vol = pd.Series([1, 2, 1, 2], name='Volume')
    avg = expected_vol.rolling(2, min_periods=1).mean()
    expected = (expected_vol >= avg * 0.5)
    pd.testing.assert_series_equal(res, expected)
