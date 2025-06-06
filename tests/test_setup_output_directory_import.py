import importlib
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

def test_setup_output_directory_is_imported():
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert hasattr(main, 'setup_output_directory')
