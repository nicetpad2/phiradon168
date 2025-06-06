import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import src.features as features


def test_get_mtf_sma_trend_invalid():
    assert features.get_mtf_sma_trend(None) == "NEUTRAL"


def test_get_mtf_sma_trend_up(monkeypatch):
    df = pd.DataFrame({'Close': [1, 2, 3]})

    def fake_sma(series, period):
        return pd.Series([1, 2, 3]) if period == 1 else pd.Series([0, 1, 2])

    monkeypatch.setattr(features, 'sma', fake_sma)
    monkeypatch.setattr(features, 'rsi', lambda s, period=14: pd.Series([50, 60, 65]))
    trend = features.get_mtf_sma_trend(df, fast=1, slow=2, rsi_period=1, rsi_upper=70)
    assert trend == 'UP'


def test_get_mtf_sma_trend_down(monkeypatch):
    df = pd.DataFrame({'Close': [3, 2, 1]})

    def fake_sma(series, period):
        return pd.Series([3, 2, 1]) if period == 1 else pd.Series([4, 3, 2])

    monkeypatch.setattr(features, 'sma', fake_sma)
    monkeypatch.setattr(features, 'rsi', lambda s, period=14: pd.Series([40, 30, 20]))
    trend = features.get_mtf_sma_trend(df, fast=1, slow=2, rsi_period=1, rsi_lower=10)
    assert trend == 'DOWN'
