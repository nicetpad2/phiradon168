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


def test_strategy_imports_shap_helpers():
    strategy = importlib.import_module('src.strategy')
    assert hasattr(strategy, 'select_top_shap_features')
    assert hasattr(strategy, 'check_model_overfit')
    assert hasattr(strategy, 'analyze_feature_importance_shap')
    assert hasattr(strategy, 'check_feature_noise_shap')
