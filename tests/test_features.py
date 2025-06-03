import pandas as pd
import numpy as np
from src import features


def test_sma_ema_basic():
    series = pd.Series([1, 2, 3, 4, 5], dtype='float32')
    sma_res = features.sma(series, 3)
    ema_res = features.ema(series, 3)
    assert np.isclose(sma_res.iloc[-1], 4.0)
    assert np.isclose(round(ema_res.iloc[-1], 3), 4.062, atol=0.01)


def test_rsi_nan_when_ta_missing(monkeypatch):
    monkeypatch.setattr(features, 'ta', None)
    series = pd.Series([1, 2, 3, 4, 5], dtype='float32')
    rsi_res = features.rsi(series, 14)
    assert rsi_res.isna().all()


def test_rolling_zscore_small():
    series = pd.Series([1, 2, 3, 4, 5], dtype='float32')
    z = features.rolling_zscore(series, 2)
    assert len(z) == 5
    assert np.isclose(z.iloc[1], 0.7071067)


def test_tag_price_structure_patterns():
    df = pd.DataFrame({
        'Gain_Z': [3, -3, 0, 0],
        'High': [1, 2, 3, 4],
        'Low': [0, 1, 2, 3],
        'Close': [1, 1.5, 2.5, 3.5],
        'Open': [0.8, 1.8, 2.2, 3.2],
        'MACD_hist': [0.5, -0.5, 0.1, -0.1],
        'Candle_Ratio': [0.6, 0.6, 0.1, 0.1],
        'Wick_Ratio': [0.1, 0.1, 0.7, 0.7],
        'Gain': [0.2, -0.2, 0.0, 0.0],
        'Candle_Body': [0.2, 0.2, 0.1, 0.1],
    })
    res = features.tag_price_structure_patterns(df)
    assert 'Pattern_Label' in res.columns
    assert "Breakout" in res["Pattern_Label"].cat.categories
