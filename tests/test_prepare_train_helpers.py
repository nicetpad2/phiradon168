import importlib
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))


def test_prepare_train_data_calls_main(monkeypatch):
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main_module = importlib.import_module('src.main')

    called = {}

    def dummy_main(run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None):
        called['run_mode'] = run_mode
        return '_ok'

    monkeypatch.setattr(main_module, 'main', dummy_main)
    main_module.prepare_train_data()
    assert called['run_mode'] == 'PREPARE_TRAIN_DATA'


def test_train_models_calls_main(monkeypatch):
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main_module = importlib.import_module('src.main')

    called = {}

    def dummy_main(run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None):
        called['run_mode'] = run_mode
        return '_ok'

    monkeypatch.setattr(main_module, 'main', dummy_main)
    main_module.train_models()
    assert called['run_mode'] == 'TRAIN_MODEL_ONLY'
