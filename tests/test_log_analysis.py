import os
import sys
import pandas as pd
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.log_analysis import parse_trade_logs, calculate_hourly_summary, calculate_position_size

SAMPLE_LOG = """
INFO:root:   Attempting to Open New Order (Standard) for SELL at 2023-01-01 10:00:00+00:00...
INFO:root:      Order Closing: Time=2023-01-01 10:10:00+00:00, Final Reason=SL, ExitPrice=1900, EntryTime=2023-01-01 10:00:00+00:00
INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=-1.0 (Raw PNL=-0.5, Comm=0.1, SpreadCost=0.2, Slip=-0.4)
INFO:root:   Attempting to Open New Order (Standard) for SELL at 2023-01-01 11:00:00+00:00...
INFO:root:      Order Closing: Time=2023-01-01 11:20:00+00:00, Final Reason=Full Close on Partial TP 1, ExitPrice=1890, EntryTime=2023-01-01 11:00:00+00:00
INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=2.5 (Raw PNL=2.8, Comm=0.1, SpreadCost=0.2, Slip=-0.2)
"""

def test_parse_trade_logs(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text(SAMPLE_LOG)
    df = parse_trade_logs(str(log_file))
    assert len(df) == 2
    assert df.iloc[0]["Reason"] == "SL"
    assert df.iloc[1]["PnL"] == 2.5

def test_calculate_hourly_summary(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text(SAMPLE_LOG)
    df = parse_trade_logs(str(log_file))
    summary = calculate_hourly_summary(df)
    assert summary.loc[10, "count"] == 1
    assert summary.loc[10, "win_rate"] == 0.0
    assert summary.loc[11, "win_rate"] == 1.0

def test_calculate_position_size():
    lot = calculate_position_size(1000, 2, 50)
    assert lot > 0
    with pytest.raises(ValueError):
        calculate_position_size(-1, 2, 50)

