import numpy as np
from pathlib import Path
from src.monitor import log_performance_metrics, monitor_auc_from_csv


def test_log_performance_metrics(tmp_path, caplog):
    y_true = [0, 1, 1, 0]
    proba = [0.2, 0.8, 0.9, 0.1]
    summary = tmp_path / "perf.csv"
    metrics = log_performance_metrics(y_true, proba, label="daily", summary_path=str(summary))
    assert summary.is_file()
    assert metrics["auc"] > 0.9
    assert metrics["accuracy"] > 0.9
    with open(summary, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert "daily" in lines[1]


def test_monitor_auc_from_csv(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("proba,target\n0.8,1\n0.1,0\n")
    summary = tmp_path / "perf.csv"
    res = monitor_auc_from_csv(str(csv_path), label="week", summary_path=str(summary))
    assert res is not None
    assert summary.is_file()
    with open(summary, "r", encoding="utf-8") as f:
        data = f.read()
    assert "week" in data


def test_monitor_auc_from_csv_missing(tmp_path):
    missing = monitor_auc_from_csv(str(tmp_path / 'none.csv'))
    assert missing is None

    bad_path = tmp_path / 'bad.csv'
    with open(bad_path, 'w', encoding='utf-8') as fh:
        fh.write('x,y\n1,2\n')
    missing_cols = monitor_auc_from_csv(str(bad_path))
    assert missing_cols is None

