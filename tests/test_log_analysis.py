import os
import sys
import pandas as pd
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.log_analysis import (
    parse_trade_logs,
    calculate_hourly_summary,
    calculate_position_size,
    calculate_reason_summary,
    calculate_duration_stats,
    calculate_drawdown_stats,
    parse_alerts,
    calculate_alert_summary,
    run_and_log_tests,
)

SAMPLE_LOG = """
INFO:root:   Attempting to Open New Order (Standard) for SELL at 2023-01-01 10:00:00+00:00...
INFO:root:      Order Closing: Time=2023-01-01 10:10:00+00:00, Final Reason=SL, ExitPrice=1900, EntryTime=2023-01-01 10:00:00+00:00
INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=-1.0 (Raw PNL=-0.5, Comm=0.1, SpreadCost=0.2, Slip=-0.4)
INFO:root:   Attempting to Open New Order (Standard) for SELL at 2023-01-01 11:00:00+00:00...
INFO:root:      Order Closing: Time=2023-01-01 11:20:00+00:00, Final Reason=Full Close on Partial TP 1, ExitPrice=1890, EntryTime=2023-01-01 11:00:00+00:00
INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=2.5 (Raw PNL=2.8, Comm=0.1, SpreadCost=0.2, Slip=-0.2)
WARNING:root:Potential issue detected
CRITICAL:root:Critical failure occurred
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


def test_additional_log_metrics(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text(SAMPLE_LOG)
    df = parse_trade_logs(str(log_file))
    reason_counts = calculate_reason_summary(df)
    assert reason_counts.loc["SL"] == 1
    duration = calculate_duration_stats(df)
    assert duration["max"] == 20
    draw_stats = calculate_drawdown_stats(df)
    assert draw_stats["total_pnl"] == pytest.approx(1.5)


def test_parse_alerts_and_summary(tmp_path):
    log_file = tmp_path / "test.log"
    log_file.write_text(SAMPLE_LOG)
    alerts = parse_alerts(str(log_file))
    assert len(alerts) == 2
    assert set(alerts["level"]) == {"WARNING", "CRITICAL"}
    summary = calculate_alert_summary(str(log_file))
    assert summary.loc["WARNING"] == 1
    assert summary.loc["CRITICAL"] == 1


def test_run_and_log_tests(tmp_path):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_sample():\n    assert True\n")
    log_output = tmp_path / "pytest.log"
    exit_code = run_and_log_tests(str(log_output), [str(test_file)])
    assert exit_code == 0
    text = log_output.read_text()
    assert "Test Run Start" in text
    assert "Test Run End" in text
    assert "1 passed" in text

