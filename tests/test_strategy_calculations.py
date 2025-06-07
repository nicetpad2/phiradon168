import os
import sys
import logging
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.strategy as strategy


def test_dynamic_tp2_multiplier_basic(monkeypatch):
    monkeypatch.setattr(strategy, 'BASE_TP_MULTIPLIER', 1.5)
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_HIGH_VOL_RATIO', 2.0)
    result = strategy.dynamic_tp2_multiplier(3.0, 1.0)
    assert result == 2.1
    result = strategy.dynamic_tp2_multiplier(1.5, 1.0)
    assert result == 1.8
    result = strategy.dynamic_tp2_multiplier(0.5, 1.0)
    assert result == 1.5
    result = strategy.dynamic_tp2_multiplier(np.nan, 1.0)
    assert result == 1.5


def test_get_adaptive_tsl_step(monkeypatch):
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_DEFAULT_STEP_R', 0.5)
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_HIGH_VOL_RATIO', 2.0)
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_HIGH_VOL_STEP_R', 1.0)
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_LOW_VOL_RATIO', 0.5)
    monkeypatch.setattr(strategy, 'ADAPTIVE_TSL_LOW_VOL_STEP_R', 0.2)
    assert strategy.get_adaptive_tsl_step(4.0, 1.0) == 1.0
    assert strategy.get_adaptive_tsl_step(0.1, 1.0) == 0.2
    assert strategy.get_adaptive_tsl_step(1.0, 1.0) == 0.5
    assert strategy.get_adaptive_tsl_step(np.nan, 1.0) == 0.5


def test_get_dynamic_signal_score_entry():
    df = pd.DataFrame({'Signal_Score': [0.2, 0.6, 0.8, 2.5]})
    assert strategy.get_dynamic_signal_score_entry(df, window=2, quantile=0.5) == 1.65
    df_empty = pd.DataFrame({'Signal_Score': []})
    assert strategy.get_dynamic_signal_score_entry(df_empty) == 0.5
    assert strategy.get_dynamic_signal_score_entry(None) == 0.5


def test_get_dynamic_signal_score_thresholds():
    s = pd.Series([0.2, 0.6, 1.2, 5.0])
    res = strategy.get_dynamic_signal_score_thresholds(s, window=2, quantile=0.5)
    assert len(res) == 4
    assert res[2] >= res[1]
    assert all(0.5 <= r <= 3.0 for r in res)


def test_adjust_lot_recovery_mode(monkeypatch, caplog):
    monkeypatch.setattr(strategy, 'RECOVERY_MODE_CONSECUTIVE_LOSSES', 3)
    monkeypatch.setattr(strategy, 'RECOVERY_MODE_LOT_MULTIPLIER', 2.0)
    monkeypatch.setattr(strategy, 'MIN_LOT_SIZE', 0.01)
    with caplog.at_level(logging.INFO):
        lot, mode = strategy.adjust_lot_recovery_mode(0.05, 4)
    assert mode == 'recovery' and lot == 0.1
    assert 'Losses' in caplog.text
    lot, mode = strategy.adjust_lot_recovery_mode(0.05, 2)
    assert mode == 'normal' and lot == 0.05


def test_calculate_aggressive_lot(monkeypatch):
    monkeypatch.setattr(strategy, 'MIN_LOT_SIZE', 0.01)
    monkeypatch.setattr(strategy, 'MAX_LOT_SIZE', 1.0)
    assert strategy.calculate_aggressive_lot(50) == 0.01
    assert strategy.calculate_aggressive_lot(600) == 0.1
    assert strategy.calculate_aggressive_lot(5000) == 1.0


def test_calculate_lot_size_fixed_risk(monkeypatch):
    monkeypatch.setattr(strategy, 'POINT_VALUE', 0.1)
    monkeypatch.setattr(strategy, 'MIN_LOT_SIZE', 0.01)
    monkeypatch.setattr(strategy, 'MAX_LOT_SIZE', 5.0)
    lot = strategy.calculate_lot_size_fixed_risk(1000, 0.01, 0.10)
    assert lot > 0.01
    lot_nan = strategy.calculate_lot_size_fixed_risk(np.nan, 0.01, 0.10)
    assert lot_nan == 0.01


def test_adjust_lot_tp2_boost(monkeypatch, caplog):
    monkeypatch.setattr(strategy, 'MIN_LOT_SIZE', 0.01)
    with caplog.at_level(logging.INFO):
        result = strategy.adjust_lot_tp2_boost(['TP', 'TP'], base_lot=0.1)
    assert result > 0.1
    assert 'TP Boost' in caplog.text
    assert strategy.adjust_lot_tp2_boost(['SL']) == 0.01


def test_calculate_lot_by_fund_mode(monkeypatch):
    monkeypatch.setattr(strategy, 'MAX_LOT_SIZE', 5.0)
    monkeypatch.setattr(strategy, 'MIN_LOT_SIZE', 0.01)
    eq = 1000
    assert strategy.calculate_lot_by_fund_mode('conservative', 0.01, eq, 1.0, 0.1) >= 0.01
    assert strategy.calculate_lot_by_fund_mode('balanced', 0.01, eq, 1.0, 0.1) >= 0.1
    assert strategy.calculate_lot_by_fund_mode('high_freq', 0.01, 50, 1.0, 0.1) == 0.01
