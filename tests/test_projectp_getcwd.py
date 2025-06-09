import importlib
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_projectp_getcwd_no_chdir_on_import(monkeypatch):
    def bad_getcwd():
        raise OSError("bad cwd")

    recorded = {}

    def fake_chdir(path):
        recorded['path'] = path

    monkeypatch.setattr(os, 'getcwd', bad_getcwd)
    monkeypatch.setattr(os, 'chdir', fake_chdir)
    monkeypatch.delitem(sys.modules, 'ProjectP', raising=False)
    importlib.import_module('ProjectP')
    assert 'path' not in recorded
