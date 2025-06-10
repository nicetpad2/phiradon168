import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def test_auto_threshold_optimization_called(monkeypatch, tmp_path):
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    main.OUTPUT_DIR = str(out_dir)

    # minimal stubs for FULL_RUN
    monkeypatch.setattr(main, 'run_all_folds_with_threshold', lambda **k: ({},{}, pd.DataFrame(), pd.DataFrame(), {}, [], None, '', '', 0.0))
    monkeypatch.setattr(main, 'select_model_for_trade', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(main, 'safe_load_csv_auto', lambda *a, **k: pd.DataFrame({'Open':[1],'High':[1],'Low':[1],'Close':[1],'datetime':['2024-01-01']}))
    monkeypatch.setattr(main, 'load_data', lambda *a, **k: pd.DataFrame({'Open':[1],'High':[1],'Low':[1],'Close':[1],'datetime':['2024-01-01']}))
    monkeypatch.setattr(main, 'prepare_datetime', lambda df, tf: df.set_index('datetime'))
    monkeypatch.setattr(main, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone':['N']}, index=df.index))
    monkeypatch.setattr(main, 'engineer_m1_features', lambda df, lag_features_config=None: df.assign(ATR_14_Shifted=0, Gain_Z=0, MACD_hist=0, MACD_hist_smooth=0, Candle_Speed=0, Pattern_Label=0, ATR_14=0, ATR_14_Rolling_Avg=0, Volatility_Index=0, ADX=0, RSI=0, Wick_Ratio=0, Candle_Body=0, Candle_Range=0, Gain=0, cluster=0, spike_score=0, session=0, model_tag=0))
    monkeypatch.setattr(main, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.1, Trade_Reason='r'))
    monkeypatch.setattr(main, 'load_features_for_model', lambda *a, **k: ['A'], raising=False)
    monkeypatch.setattr(main, 'maybe_collect', lambda: None)
    monkeypatch.setattr(main.sys, 'exit', lambda *a, **k: None)
    class CatBoostClassifier:
        def predict_proba(self, X):
            return [0]
    monkeypatch.setattr(main, 'load', lambda *a, **k: CatBoostClassifier(), raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)
    monkeypatch.setattr(main, 'ENABLE_OPTUNA_TUNING', False, raising=False)
    monkeypatch.setattr(main, 'USE_GPU_ACCELERATION', True, raising=False)
    class NVMLStub:
        def nvmlShutdown(self):
            pass
    monkeypatch.setattr(main, 'pynvml', NVMLStub(), raising=False)
    monkeypatch.setattr(main, 'nvml_handle', object(), raising=False)
    monkeypatch.setattr(main, 'print_gpu_utilization', lambda *a, **k: None, raising=False)

    import src.features as feats
    monkeypatch.setattr(main, 'ENABLE_AUTO_THRESHOLD_TUNING', True, raising=False)
    monkeypatch.setattr(feats, 'ENABLE_AUTO_THRESHOLD_TUNING', True, raising=False)
    captured = {}
    import threshold_optimization as topt
    monkeypatch.setattr(topt, 'run_threshold_optimization', lambda **kw: captured.setdefault('args', kw))

    main.run_auto_threshold_stage()

    assert captured.get('args') == {
        'output_dir': main.OUTPUT_DIR,
        'trials': main.OPTUNA_N_TRIALS,
        'study_name': 'threshold_wfv',
        'direction': main.OPTUNA_DIRECTION,
        'timeout': None,
    }
