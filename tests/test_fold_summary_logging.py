import os, sys
import pandas as pd
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def test_fold_summary_logging(simple_m1_df, caplog):
    df = simple_m1_df.copy()
    df['ATR_14_Shifted'] = 0.1
    df['ATR_14'] = 0.1
    df['ATR_14_Rolling_Avg'] = 0.1
    required_extra = [
        'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed',
        'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score',
        'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 'Wick_Ratio',
        'Candle_Body', 'Candle_Range', 'Gain', 'cluster', 'spike_score', 'session'
    ]
    for col in required_extra:
        df[col] = 0
    df['Entry_Long'] = 1
    df.index.name = 'Datetime'

    with caplog.at_level(logging.WARNING):
        strategy.run_backtest_simulation_v34(
            df,
            label='TEST',
            initial_capital_segment=1000.0,
            side='BUY',
            fund_profile={'mm_mode': 'balanced', 'risk': 0.01},
            fold_config={},
            current_fold_index=0,
        )

    assert any('[QA][SUMMARY] Fold Finished' in m for m in caplog.messages)
