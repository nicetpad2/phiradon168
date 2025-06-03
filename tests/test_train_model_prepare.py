import os
from pathlib import Path
import pandas as pd
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def test_train_model_only_auto_prepares(monkeypatch, tmp_path):
    # Setup output directory globals
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = "out"
    if hasattr(main, 'OUTPUT_DIR'):
        main.OUTPUT_DIR = ""

    # Dummy prepare_train_data that creates required csv files
    def dummy_prepare():
        main.OUTPUT_DIR = os.path.join(main.OUTPUT_BASE_DIR, main.OUTPUT_DIR_NAME)
        os.makedirs(main.OUTPUT_DIR, exist_ok=True)
        (Path(main.OUTPUT_DIR) / 'trade_log_v32_walkforward_prep_data_NORMAL.csv').write_text('entry_time\n2023-01-01')
        (Path(main.OUTPUT_DIR) / 'final_data_m1_v32_walkforward_prep_data_NORMAL.csv').write_text('Open\n1')
        return '_prep'

    monkeypatch.setattr(main, 'prepare_train_data', dummy_prepare)
    monkeypatch.setattr(main, 'safe_load_csv_auto', lambda p: pd.read_csv(p))
    monkeypatch.setattr(main, 'train_and_export_meta_model', lambda **k: ({'main': 'm.pkl'}, ['X']))
    monkeypatch.setattr(main, 'ENABLE_OPTUNA_TUNING', False, raising=False)
    monkeypatch.setattr(main, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(main, 'pynvml', None, raising=False)
    monkeypatch.setattr(main, 'nvml_handle', None, raising=False)

    result = main.main(run_mode='TRAIN_MODEL_ONLY')
    assert result == '_train_only'
