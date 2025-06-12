import pandas as pd
import logging
from src import strategy
import pytest


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


def test_run_backtest_simulation_missing_price_cols(simple_m1_df):
    df = simple_m1_df.drop(columns=['High'])
    df['ATR_14_Shifted'] = 1.0
    df['ATR_14'] = 1.0
    with pytest.raises(ValueError):
        strategy.run_backtest_simulation_v34(
            df,
            label='TEST',
            initial_capital_segment=1000,
            fold_config={},
            current_fold_index=0,
        )


def test_run_backtest_simulation_ml_feature_check(simple_m1_df, monkeypatch):
    df = simple_m1_df.copy()
    df['ATR_14_Shifted'] = 0.1
    df['ATR_14'] = 0.1
    df['ATR_14_Rolling_Avg'] = 0.1
    required_extra = [
        'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed',
        'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag',
        'Signal_Score', 'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI',
        'Wick_Ratio', 'Candle_Body', 'Candle_Range', 'Gain', 'cluster',
        'spike_score', 'session'
    ]
    for col in required_extra:
        df[col] = 0
    df['Entry_Long'] = 1
    missing_feature = 'dummy_feature'

    class DummyModel:
        def predict_proba(self, X):
            return [[0.6, 0.4]]

    available_models = {
        'main': {'model': DummyModel(), 'features': ['Signal_Score', missing_feature]}
    }
    monkeypatch.setattr(strategy, 'USE_META_CLASSIFIER', True, raising=False)
    result = strategy.run_backtest_simulation_v34(
        df,
        label='TEST',
        initial_capital_segment=1000,
        side='BUY',
        fund_profile={'mm_mode': 'balanced', 'risk': 0.01},
        fold_config={},
        current_fold_index=0,
        available_models=available_models,
        model_switcher_func=lambda ctx, models: ('main', 1.0)
    )
    run_summary = result[5]
    assert run_summary['error_in_loop'] is False


def test_run_backtest_invalid_index(simple_m1_df, caplog):
    df = simple_m1_df.reset_index(drop=True)
    df['ATR_14_Shifted'] = 1.0
    df['ATR_14'] = 1.0
    df['ATR_14_Rolling_Avg'] = 0.1
    required_extra = [
        'Trend_Zone', 'Gain_Z', 'MACD_hist', 'MACD_hist_smooth', 'Candle_Speed',
        'Pattern_Label', 'Entry_Long', 'Entry_Short', 'Trade_Tag',
        'Signal_Score', 'Trade_Reason', 'Volatility_Index', 'ADX', 'RSI',
        'Wick_Ratio', 'Candle_Body', 'Candle_Range', 'Gain', 'cluster',
        'spike_score', 'session'
    ]
    for col in required_extra:
        df[col] = 0
    with caplog.at_level(logging.ERROR):
        result = strategy.run_backtest_simulation_v34(
            df,
            label='TEST',
            initial_capital_segment=1000,
            fold_config={},
            current_fold_index=0,
        )
    run_summary = result[5]
    assert run_summary['error_in_loop']
    assert 'index is not DatetimeIndex' in caplog.text
