import importlib
import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_optuna_stub_raises_importerror():
    if 'optuna' in sys.modules:
        del sys.modules['optuna']
    optuna = importlib.import_module('optuna')
    assert hasattr(optuna, 'Trial')
    with pytest.raises(ImportError):
        optuna.create_study()
