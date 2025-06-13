import json
import pandas as pd
import pytest

import wfv_orchestrator as wfv_orch
from src.wfv_aggregator import aggregate_wfv_results


def _create_fold(tmpdir, idx, pnl):
    d = tmpdir / f"fold_{idx}"
    d.mkdir()
    df = pd.DataFrame({"entry_time": [f"2025-01-0{idx+1}"], "pnl_usd_net": [pnl]})
    df.to_csv(d / "oos_trade_log.csv", index=False)
    with open(d / "oos_summary.json", "w", encoding="utf-8") as f:
        json.dump({"total_net_profit": pnl}, f)


def test_aggregate_wfv_results(tmp_path):
    _create_fold(tmp_path, 0, 1.0)
    _create_fold(tmp_path, 1, -0.5)
    combined = aggregate_wfv_results(str(tmp_path))
    assert len(combined) == 2
    summary_file = tmp_path / "aggregated_summary.json"
    assert summary_file.exists()
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["Total Net Profit"] == pytest.approx(0.5)
    assert "Win Rate" in summary


def test_aggregate_wfv_results_no_folds(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_wfv_results(str(tmp_path))


def test_run_wfv_simple(tmp_path):
    data = pd.DataFrame({"Close": [1, 2, 3, 4, 5, 6]})
    wfv_orch.run_wfv_simple(data, str(tmp_path), n_splits=2)
    assert (tmp_path / "full_oos_trade_log.csv").exists()
    with open(tmp_path / "aggregated_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "Total Net Profit" in summary
