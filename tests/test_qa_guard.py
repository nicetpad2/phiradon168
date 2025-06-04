import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def test_trade_log_created_even_if_empty(monkeypatch, tmp_path):
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    if hasattr(main, 'OUTPUT_DIR'):
        main.OUTPUT_DIR = ''

    def dummy_run_all_folds_with_threshold(**kwargs):
        return ({'ok': 1}, {'ok': 1}, pd.DataFrame(), pd.DataFrame(), {}, [], None, 'L1', 'L2', 0.0)

    monkeypatch.setattr(main, 'run_all_folds_with_threshold', dummy_run_all_folds_with_threshold)
    monkeypatch.setattr(main, 'select_model_for_trade', lambda *a, **k: None, raising=False)
    monkeypatch.setattr(main, 'load_features_for_model', lambda *a, **k: [], raising=False)
    class CatBoostClassifier:
        def predict_proba(self, X):  # pragma: no cover - simple stub
            return [0]

    monkeypatch.setattr(main, 'load', lambda *a, **k: CatBoostClassifier(), raising=False)
    monkeypatch.setattr(os.path, 'exists', lambda p: True)
    monkeypatch.setattr(main, 'ENABLE_OPTUNA_TUNING', False, raising=False)
    monkeypatch.setattr(main, 'MULTI_FUND_MODE', False, raising=False)
    monkeypatch.setattr(main, 'FUND_PROFILES', {'DEF': {'risk':1, 'mm_mode':'static'}}, raising=False)
    monkeypatch.setattr(main, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(main, 'pynvml', None, raising=False)
    monkeypatch.setattr(main, 'nvml_handle', None, raising=False)

    suffix = main.main(run_mode='FULL_RUN')
    out_dir = Path(main.OUTPUT_DIR)
    log_gz = out_dir / f'trade_log_v32_walkforward{suffix}.csv.gz'
    log_csv = out_dir / f'trade_log_v32_walkforward{suffix}.csv'
    assert log_gz.exists() or log_csv.exists()
    qa_log = out_dir / '.qa.log'
    assert qa_log.exists()
    # New check for simplified trade log generated via export_trade_log
    assert (out_dir / 'trade_log_NORMAL.csv').exists()
