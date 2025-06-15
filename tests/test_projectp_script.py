import runpy
import types
import sys
import pytest


def test_script_calls_main(monkeypatch):
    called = {}
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(main=lambda: called.setdefault("run", True)))
    runpy.run_path("ProjectP.py", run_name="__main__")
    assert called.get("run") is True


def test_script_main_returns_zero(monkeypatch):
    called = {}
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(main=lambda: (called.setdefault("run", True), 0)[1]))
    import ProjectP
    result = ProjectP._script_main()
    assert called.get("run") is True
    assert result == 0
