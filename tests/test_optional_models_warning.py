import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def test_optional_models_warning(monkeypatch, tmp_path, caplog):
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    if hasattr(main, 'OUTPUT_DIR'):
        main.OUTPUT_DIR = ''

    out_dir = Path(main.OUTPUT_BASE_DIR) / main.OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'meta_classifier.pkl', 'w') as f:
        f.write('x')
    with open(out_dir / 'features_main.json', 'w', encoding='utf-8') as f:
        json.dump([], f)

    class DummyModel:
        def predict_proba(self, X):
            return [0]

    monkeypatch.setattr(main, 'run_all_folds_with_threshold', lambda **kw: ({}, {}, pd.DataFrame(), pd.DataFrame(), {}, [], None, '', '', 0.0), raising=False)
    monkeypatch.setattr(main, 'select_model_for_trade', lambda *a, **k: 'main', raising=False)
    monkeypatch.setattr(main, 'load_features_for_model', lambda *a, **k: [], raising=False)
    monkeypatch.setattr(main, 'load', lambda p: DummyModel(), raising=False)
    monkeypatch.setattr(main, 'USE_GPU_ACCELERATION', False, raising=False)
    monkeypatch.setattr(main, 'MULTI_FUND_MODE', False, raising=False)
    monkeypatch.setattr(main, 'FUND_PROFILES', {'DEF': {'risk':1, 'mm_mode':'static'}}, raising=False)
    monkeypatch.setattr(main, 'pynvml', None, raising=False)
    monkeypatch.setattr(main, 'nvml_handle', None, raising=False)

    caplog.set_level(logging.WARNING)
    main.main(run_mode='FULL_RUN')
    assert "ไม่พบไฟล์ Model 'spike'" in caplog.text
    assert "ไม่พบไฟล์ Model 'cluster'" in caplog.text
    assert "ไม่พบไฟล์ Model 'main'" not in caplog.text
