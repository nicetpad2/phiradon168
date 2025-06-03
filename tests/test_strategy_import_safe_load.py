import importlib
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

def test_strategy_has_safe_load_csv_auto():
    strategy = importlib.import_module('src.strategy')
    assert hasattr(strategy, 'safe_load_csv_auto')

def test_strategy_has_simple_converter():
    strategy = importlib.import_module('src.strategy')
    assert hasattr(strategy, 'simple_converter')
