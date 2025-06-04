import os
import importlib

# Re-import src.main to ensure new directory function executed
if 'src.main' in importlib.sys.modules:  # pragma: no cover - import cleanup
    del importlib.sys.modules['src.main']

import src.main as main

def test_ensure_default_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DEFAULT_OUTPUT_DIR', str(tmp_path/'out'), raising=False)
    path = main.ensure_default_output_dir(main.DEFAULT_OUTPUT_DIR)
    assert os.path.isdir(path)
