import importlib
import sys
import types
import os

import logging


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_fund_profiles_loaded_from_config(monkeypatch):
    dummy = types.ModuleType('config')
    dummy.ENTRY_CONFIG_PER_FOLD = {}
    dummy.FUND_PROFILES = {'X': {'risk': 1, 'mm_mode': 'demo'}}
    dummy.MULTI_FUND_MODE = True
    dummy.DEFAULT_FUND_NAME = 'X'
    dummy.logger = logging.getLogger('x')
    dummy.print_gpu_utilization = lambda *_: None
    monkeypatch.setitem(sys.modules, 'src.config', dummy)
    if 'src' in sys.modules:
        monkeypatch.setattr(sys.modules['src'], 'config', dummy, raising=False)
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert main.DEFAULT_FUND_PROFILES == {'X': {'risk': 1, 'mm_mode': 'demo'}}
    assert main.DEFAULT_MULTI_FUND_MODE is True
    assert main.DEFAULT_FUND_NAME == 'X'
