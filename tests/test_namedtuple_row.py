from collections import namedtuple
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.strategy import is_entry_allowed

def test_is_entry_allowed_namedtuple():
    Bar = namedtuple('Bar', ['spike_score','ADX','Wick_Ratio','Volatility_Index','Candle_Body','Candle_Range','Gain','ATR_14','Signal_Score','Candle_Ratio','Trend_Zone'])
    row = Bar(0.1, 25, 0.5, 1.2, 0.5, 1.0, 1.0, 5.0, 2.5, 0.2, 'UP')
    allowed, reason = is_entry_allowed(row, 'London', 0, 'BUY', row.Trend_Zone, signal_score_threshold=1.0)
    assert allowed is True
    assert reason == 'ALLOWED'

    row_block = Bar(0.1, 25, 0.5, 1.2, 0.5, 1.0, 1.0, 5.0, 2.5, 0.2, 'DOWN')
    allowed, reason = is_entry_allowed(row_block, 'London', 0, 'BUY', row_block.Trend_Zone, signal_score_threshold=1.0)
    assert allowed is False
    assert 'M15_TREND' in reason
