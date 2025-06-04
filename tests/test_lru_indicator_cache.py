import pandas as pd
import src.features as features


def test_calculate_sma_lru_cache():
    prices = tuple(range(20))
    res1 = features.calculate_sma('XAUUSD', 'M1', 5, '2020-01-01', prices)
    res2 = features.calculate_sma('XAUUSD', 'M1', 5, '2020-01-01', prices)
    assert res1 is res2


def test_calculate_rsi_lru_cache(monkeypatch):
    prices = tuple(range(20))
    monkeypatch.setattr(features, 'ta', None)
    res1 = features.calculate_rsi('XAUUSD', 'M1', 14, '2020-01-01', prices)
    res2 = features.calculate_rsi('XAUUSD', 'M1', 14, '2020-01-01', prices)
    assert res1 is res2
