import importlib
import os
import sys
import gc
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

MODULE_PATH = 'src.utils.gc_utils'


def reload_module(value):
    """Reload module with ENV var value."""
    if value is None:
        os.environ.pop("ENABLE_MANUAL_GC", None)
    else:
        os.environ["ENABLE_MANUAL_GC"] = value
    if MODULE_PATH in list(sys.modules):
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


@pytest.mark.parametrize('value', ['1', 'True', 'true'])
def test_collect_called(monkeypatch, value):
    module = reload_module(value)
    called = []
    monkeypatch.setattr(gc, 'collect', lambda: called.append(True))
    module.maybe_collect()
    assert called == [True]


@pytest.mark.parametrize('value', ['0', 'False', '', None])
def test_collect_not_called(monkeypatch, value):
    module = reload_module(value)
    called = []
    monkeypatch.setattr(gc, 'collect', lambda: called.append(True))
    module.maybe_collect()
    assert called == []
