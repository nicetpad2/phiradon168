import os, sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import strategy, log_analysis


def test_run_all_folds_handles_no_trades(simple_m1_df, monkeypatch, tmp_path):
    def dummy_run(*args, **kwargs):
        return (
            simple_m1_df.iloc[:0],
            pd.DataFrame(),
            1000.0,
            {},
            0.0,
            {},
            [],
            'L1',
            'L2',
            False,
            0,
            0.0,
        )

    monkeypatch.setattr(strategy, 'run_backtest_simulation_v34', dummy_run)

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    df = simple_m1_df.copy()
    df['ATR_14_Shifted'] = 0.1
    df['ATR_14'] = 0.1
    df['ATR_14_Rolling_Avg'] = 0.1

    result = strategy.run_all_folds_with_threshold(
        fund_profile={'name': 'T', 'mm_mode': 'static', 'risk': 1},
        df_m1_final=df,
        n_walk_forward_splits=2,
        output_dir=str(out_dir)
    )

    metrics_buy, metrics_sell, df_final, trade_log = result[0], result[1], result[2], result[3]
    assert isinstance(metrics_buy, dict)
    assert trade_log.empty


def test_summarize_block_reasons():
    logs = [
        {"reason": "ML_META_FILTER"},
        {"reason": "ML_META_FILTER"},
        {"reason": "SOFT_COOLDOWN"},
    ]
    result = log_analysis.summarize_block_reasons(logs)
    assert result["ML_META_FILTER"] == 2
    assert result["SOFT_COOLDOWN"] == 1
