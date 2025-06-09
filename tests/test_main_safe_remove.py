import logging
from pathlib import Path
import sys
import types
sys.modules.setdefault("torch", types.SimpleNamespace())
import src.main as main
import src.config as cfg

def _setup_dirs(tmp_path, inside_data):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    if inside_data:
        out_dir = data_dir / 'out'
    else:
        out_dir = tmp_path / 'out'
    out_dir.mkdir()
    main.OUTPUT_BASE_DIR = str(tmp_path)
    main.OUTPUT_DIR_NAME = 'out'
    main.OUTPUT_DIR = str(out_dir)
    cfg.DATA_DIR = data_dir
    # create existing target files
    (out_dir / 'trade_log_v32_walkforward.csv.gz').write_text('old')
    (out_dir / 'final_data_m1_v32_walkforward.csv.gz').write_text('old')
    return out_dir

def _patch_pipeline(monkeypatch, out_dir):
    def fake_prepare():
        log_gz = out_dir / 'trade_log_v32_walkforward_prep_data_OK.csv.gz'
        data_gz = out_dir / 'final_data_m1_v32_walkforward_prep_data_OK.csv.gz'
        log_gz.write_text('newlog')
        data_gz.write_text('newdata')
        return '_prep_data_OK'
    monkeypatch.setattr(main, 'prepare_train_data', fake_prepare)
    monkeypatch.setattr(main, 'train_models', lambda: None)
    monkeypatch.setattr(main.glob, 'glob', lambda pattern: ['x'] if 'meta_classifier' in pattern else [])
    monkeypatch.setattr(main.shutil, 'move', lambda src, dst: Path(src).rename(dst))

def test_safe_removal_outside_data_dir(monkeypatch, tmp_path, caplog):
    out_dir = _setup_dirs(tmp_path, inside_data=False)
    _patch_pipeline(monkeypatch, out_dir)
    removed = []
    monkeypatch.setattr(main.os, 'remove', lambda p: removed.append(str(p)))
    called = {}
    orig_main = main.main
    monkeypatch.setattr(main, 'main', lambda run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None: called.setdefault('mode', run_mode) or '_ok')
    caplog.set_level(logging.WARNING)
    result = orig_main(run_mode='FULL_PIPELINE')
    assert called['mode'] == 'FULL_RUN'
    assert result == '_ok'
    assert removed == []
    assert 'outside DATA_DIR' in caplog.text

def test_safe_removal_inside_data_dir(monkeypatch, tmp_path, caplog):
    out_dir = _setup_dirs(tmp_path, inside_data=True)
    _patch_pipeline(monkeypatch, out_dir)
    removed = []
    monkeypatch.setattr(main.os, 'remove', lambda p: removed.append(str(p)))
    called = {}
    orig_main = main.main
    monkeypatch.setattr(main, 'main', lambda run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None: called.setdefault('mode', run_mode) or '_ok')
    caplog.set_level(logging.WARNING)
    result = orig_main(run_mode='FULL_PIPELINE')
    assert called['mode'] == 'FULL_RUN'
    assert result == '_ok'
    trade = str(Path(out_dir, 'trade_log_v32_walkforward.csv.gz'))
    data = str(Path(out_dir, 'final_data_m1_v32_walkforward.csv.gz'))
    assert trade in removed and data in removed
    assert 'outside DATA_DIR' not in caplog.text
