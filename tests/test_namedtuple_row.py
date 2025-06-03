from collections import namedtuple
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.strategy import is_entry_allowed

def test_is_entry_allowed_namedtuple():
    Bar = namedtuple('Bar', ['spike_score','ADX','Wick_Ratio','Volatility_Index','Candle_Body','Candle_Range','Gain','ATR_14','Signal_Score','Candle_Ratio'])
    row = Bar(0.1, 25, 0.5, 0.7, 0.5, 1.0, 1.0, 5.0, 2.5, 0.2)
    allowed, reason = is_entry_allowed(row, 'London', 0, signal_score_threshold=1.0)
    assert allowed is True
    assert reason == 'ALLOWED'
