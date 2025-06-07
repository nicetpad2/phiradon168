import os
import sys
import logging
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.strategy as strategy


def test_summarize_wfv_results_basic():
    metrics = [
        {
            "buy": {"Fold 1 Buy (": 10},
            "sell": {"Fold 1 Sell (": 5},
        },
        {
            "buy": {"Fold 2 Buy (": 20},
            "sell": {"Fold 2 Sell (": -10},
        },
    ]
    df = strategy.summarize_wfv_results(metrics)
    assert list(df.columns) == ["fold_no", "PnL_total", "Win_Rate", "Max_Drawdown"]
    assert df.iloc[0].to_dict() == {"fold_no": 1, "PnL_total": 10, "Win_Rate": 10, "Max_Drawdown": 10}
    assert df.iloc[1].to_dict() == {"fold_no": 2, "PnL_total": 20, "Win_Rate": 20, "Max_Drawdown": 20}


def test_adjust_gain_z_threshold_by_drift(monkeypatch, caplog):
    monkeypatch.setattr(strategy, 'DYNAMIC_GAINZ_DRIFT_THRESHOLD', 0.05, raising=False)
    monkeypatch.setattr(strategy, 'DYNAMIC_GAINZ_ADJUSTMENT', 0.02, raising=False)
    with caplog.at_level(logging.INFO):
        res = strategy.adjust_gain_z_threshold_by_drift({"Gain_Z": {"wasserstein": 0.1}}, 0.5)
    assert res == 0.52
    assert "Adjusting GainZ Threshold" in caplog.text
    assert strategy.adjust_gain_z_threshold_by_drift({"Gain_Z": {"wasserstein": 0.01}}, 0.5) == 0.5
    assert strategy.adjust_gain_z_threshold_by_drift(None, 0.5) == 0.5
    assert strategy.adjust_gain_z_threshold_by_drift({}, "0.5") == "0.5"
