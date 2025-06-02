import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_stub_functions_present():
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert hasattr(main, 'setup_fonts')
    assert hasattr(main, 'print_gpu_utilization')
