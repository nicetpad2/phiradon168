import pandas as pd
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.log_analysis import (
    calculate_hourly_summary,
    calculate_reason_summary,
    calculate_duration_stats,
    calculate_drawdown_stats,
    calculate_expectancy,
    calculate_alert_summary,
    compile_log_summary,
    summarize_block_reasons,
    plot_expectancy_by_period,
)


def test_summary_functions_on_empty_df(tmp_path):
    empty_df = pd.DataFrame()
    summary = calculate_hourly_summary(empty_df)
    assert list(summary.columns) == ["count", "win_rate", "avg_pnl"]
    assert summary.empty

    reasons = calculate_reason_summary(empty_df)
    assert reasons.empty

    duration = calculate_duration_stats(empty_df)
    assert duration == {"mean": 0.0, "median": 0.0, "max": 0.0}

    stats = calculate_drawdown_stats(empty_df)
    assert stats == {"total_pnl": 0.0, "max_drawdown": 0.0}

    exp = calculate_expectancy(empty_df)
    assert exp == 0.0


def test_calculate_expectancy_nan_handling():
    df = pd.DataFrame({"PnL": [1.0, None, float('nan')]})
    # Only the first value is valid so expectancy should equal that value
    assert calculate_expectancy(df) == 1.0


def test_alert_summary_and_compile(tmp_path):
    log = tmp_path / "alerts.log"
    log.write_text("INFO:ok\n")
    summary = calculate_alert_summary(str(log))
    assert summary.empty

    df = pd.DataFrame({"EntryTime": [], "CloseTime": [], "PnL": []})
    comp = compile_log_summary(df, str(log))
    assert set(comp.keys()) == {"hourly", "reasons", "duration", "pnl", "alerts"}
    assert comp["alerts"].empty


def test_summarize_block_reasons():
    assert summarize_block_reasons([]).empty
    logs = [
        {"reason": "A"},
        {"reason": "B"},
        {"reason": "A"},
        "bad",
    ]
    counts = summarize_block_reasons(logs)
    assert counts.loc["A"] == 2
    assert counts.loc["B"] == 1


def test_calculate_expectancy_all_nan():
    df = pd.DataFrame({"PnL": [float('nan'), None]})
    assert calculate_expectancy(df) == 0.0


def test_plot_expectancy_by_period():
    exp = pd.Series([0.1, -0.2], index=['a', 'b'])
    fig = plot_expectancy_by_period(exp)
    assert hasattr(fig, 'savefig')
