import runpy
import types
import sys
import pytest


def test_script_calls_main(monkeypatch):
    called = {}
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(main=lambda: called.setdefault("run", True)))
    with pytest.raises(SystemExit):
        runpy.run_path("ProjectP.py", run_name="__main__")
    assert called.get("run") is True
