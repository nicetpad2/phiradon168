import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def make_df():
    df = pd.read_csv('tests/fixtures/sample_data.csv', index_col=0, parse_dates=True)
    extras = [
        'ATR_14_Shifted', 'ATR_14_Rolling_Avg', 'Trend_Zone', 'Gain_Z',
        'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed', 'Pattern_Label',
        'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score',
        'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 'Wick_Ratio',
        'Candle_Body', 'Candle_Range', 'Gain', 'cluster', 'spike_score', 'session'
    ]
    for col in extras:
        df[col] = 0
    df['ATR_14_Shifted'] = df['ATR_14']
    df['ATR_14_Rolling_Avg'] = df['ATR_14']
    df['Entry_Long'] = 1
    return df


def test_smoke_backtest():
    df = make_df().head(10)
    result = strategy.run_backtest_simulation_v34(
        df,
        label='SMOKE',
        initial_capital_segment=1000,
        fold_config={},
        current_fold_index=0,
        fund_profile={'mm_mode': 'balanced', 'risk': 0.01}
    )
    run_summary = result[5]
    assert 'error_in_loop' in run_summary
