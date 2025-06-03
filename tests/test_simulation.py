import pandas as pd
from src import strategy


def test_run_backtest_simulation_missing_cols(simple_m1_df):
    df = simple_m1_df.copy()
    df['ATR_14_Shifted'] = 1.0
    df['ATR_14'] = 1.0
    # omit ATR_14_Rolling_Avg to trigger early return
    required_extra = ['Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed',
                      'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score',
                      'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI', 'Wick_Ratio', 'Candle_Body',
                      'Candle_Range', 'Gain', 'cluster', 'spike_score', 'session']
    for col in required_extra:
        df[col] = 0
    result = strategy.run_backtest_simulation_v34(
        df,
        label='TEST',
        initial_capital_segment=1000,
        side='BUY',
        fund_profile={'mm_mode': 'balanced', 'risk': 0.01},
        fold_config={},
        current_fold_index=0
    )
    run_summary = result[5]
    assert run_summary['error_in_loop']
