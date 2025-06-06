import os, sys
import pandas as pd
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def test_forced_entry_no_unbound_error(simple_m1_df, monkeypatch, caplog):
    df = simple_m1_df.copy()
    # Add minimal required columns
    df['ATR_14'] = 1.0
    df['ATR_14_Shifted'] = 1.0
    df['ATR_14_Rolling_Avg'] = 1.0
    df['Gain_Z'] = 1.0
    df['Pattern_Label'] = 'Breakout'
    df['Signal_Score'] = 1.0
    df['session'] = 'Asia'
    df['Trend_Zone'] = 'UP'
    df['MACD_hist'] = 0.0
    df['MACD_hist_smooth'] = 0.0
    df['Candle_Speed'] = 0.0
    df['Trade_Reason'] = 'test'
    df['Entry_Long'] = 0
    df['Entry_Short'] = 0
    df['Trade_Tag'] = 'test'
    df['Volatility_Index'] = 0.0
    df['ADX'] = 0.0
    df['RSI'] = 0.0
    df['Wick_Ratio'] = 0.0
    df['Candle_Body'] = 0.0
    df['Candle_Range'] = 0.0
    df['Gain'] = 0.0
    df['cluster'] = 0
    df['spike_score'] = 0.0

    # Patch constants to trigger forced entry quickly
    monkeypatch.setattr(strategy, 'ENABLE_FORCED_ENTRY', True)
    monkeypatch.setattr(strategy, 'FORCED_ENTRY_BAR_THRESHOLD', 1)
    monkeypatch.setattr(strategy, 'FORCED_ENTRY_MIN_SIGNAL_SCORE', 0.5)
    monkeypatch.setattr(strategy, 'FORCED_ENTRY_CHECK_MARKET_COND', False)

    with caplog.at_level(logging.INFO):
        strategy.run_backtest_simulation_v34(
            df,
            label='TEST',
            initial_capital_segment=100.0,
            side='BUY',
            fund_profile={'mm_mode': 'balanced', 'risk': 0.01},
            fold_config={},
            current_fold_index=0,
        )
    assert 'Attempt Forced BUY' in caplog.text or 'Attempting to Open Forced Order' in caplog.text
