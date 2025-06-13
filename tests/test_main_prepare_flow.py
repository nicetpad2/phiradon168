import os
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def test_main_prepare_train_data_flow(monkeypatch, tmp_path):
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    if hasattr(main, 'OUTPUT_DIR'):
        main.OUTPUT_DIR = ''

    dates = pd.date_range('2024-01-01', periods=3, freq='min')
    df_base = pd.DataFrame({
        'Open': [1.0, 1.1, 1.2],
        'High': [1.0, 1.1, 1.2],
        'Low': [1.0, 1.1, 1.2],
        'Close': [1.0, 1.1, 1.2],
        'datetime': dates
    })

    monkeypatch.setattr(main, 'load_data', lambda p, tf, dtypes=None: df_base.copy())
    monkeypatch.setattr(main, 'load_validated_csv', lambda p, label, dtypes=None: df_base.copy())
    monkeypatch.setattr(main, 'prepare_datetime', lambda df, tf: df.set_index('datetime'))
    monkeypatch.setattr(main, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['NEUTRAL'] * len(df)}, index=df.index))

    def fake_engineer(df, lag_features_config=None):
        df = df.copy()
        df['ATR_14_Shifted'] = 0.1
        df['Gain_Z'] = 0.1
        df['MACD_hist'] = 0.1
        df['MACD_hist_smooth'] = 0.1
        df['Candle_Speed'] = 0.1
        df['Pattern_Label'] = 0
        df['ATR_14'] = 0.1
        df['ATR_14_Rolling_Avg'] = 0.1
        df['Volatility_Index'] = 0.1
        df['ADX'] = 0.1
        df['RSI'] = 50
        df['Wick_Ratio'] = 0.1
        df['Candle_Body'] = 0.1
        df['Candle_Range'] = 0.1
        df['Gain'] = 0.1
        df['cluster'] = 0
        df['spike_score'] = 0.0
        df['session'] = 0
        df['model_tag'] = 0
        return df

    monkeypatch.setattr(main, 'engineer_m1_features', fake_engineer)
    monkeypatch.setattr(main, 'clean_m1_data', lambda df: (df, list(df.columns)))
    monkeypatch.setattr(main, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.1, Trade_Reason='r'))
    monkeypatch.setattr(main, 'load_features_for_model', lambda name, out_dir: None)
    monkeypatch.setattr(main, 'run_all_folds_with_threshold', lambda **kw: ({}, {}, pd.DataFrame(), pd.DataFrame(), {}, [], None, '', '', 0.0))
    monkeypatch.setattr(main, 'maybe_collect', lambda: None)

    suffix = main.main(run_mode='PREPARE_TRAIN_DATA')
    assert suffix == f'_prep_data_{main.DEFAULT_FUND_NAME}'

    out_dir = Path(main.OUTPUT_DIR)
    assert (out_dir / f'final_data_m1_v32_walkforward{suffix}.csv.gz').exists()
    assert (out_dir / f'trade_log_v32_walkforward{suffix}.csv.gz').exists()
    assert (out_dir / 'features_main.json').exists()


def test_main_full_pipeline_simple(monkeypatch, tmp_path):
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    main.OUTPUT_DIR = str(out_dir)

    (out_dir / 'trade_log_v32_walkforward.csv.gz').write_text('x')
    (out_dir / 'meta_classifier.pkl').write_text('x')

    monkeypatch.setattr(main.glob, 'glob', lambda pattern: [str(out_dir / 'meta_classifier.pkl')] if 'meta_classifier' in pattern else [])

    called = {}
    orig_main = main.main

    def dummy_full_run(run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None):
        called['mode'] = run_mode
        return '_ok'

    monkeypatch.setattr(main, 'main', dummy_full_run)

    result = orig_main(run_mode='FULL_PIPELINE')
    assert called['mode'] == 'FULL_RUN'
    assert result == '_ok'
