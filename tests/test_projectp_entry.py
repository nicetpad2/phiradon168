import importlib
import runpy
import types
import sys
import pytest


def test_import_exposes_main():
    mod = importlib.import_module("ProjectP")
    assert hasattr(mod, "main")


def test_script_invokes_main(monkeypatch):
    called = {}
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(main=lambda: called.setdefault("run", True)))
    runpy.run_path("ProjectP.py", run_name="__main__")
    assert called.get("run") is True
