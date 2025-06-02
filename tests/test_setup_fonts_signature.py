import inspect
import importlib
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_setup_fonts_zero_args():
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    sig = inspect.signature(main.setup_fonts)
    assert len(sig.parameters) == 0

