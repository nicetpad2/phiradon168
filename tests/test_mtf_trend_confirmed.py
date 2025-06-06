import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.strategy import is_mtf_trend_confirmed


def test_mtf_trend_accepts_neutral():
    assert is_mtf_trend_confirmed('NEUTRAL', 'BUY') is True
    assert is_mtf_trend_confirmed('NEUTRAL', 'SELL') is True
    assert is_mtf_trend_confirmed('UP', 'BUY') is True
    assert is_mtf_trend_confirmed('DOWN', 'SELL') is True
    assert not is_mtf_trend_confirmed('DOWN', 'BUY')
    assert not is_mtf_trend_confirmed('UP', 'SELL')
