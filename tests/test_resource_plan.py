import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import json
import builtins
import types
import logging

import pytest

from src.utils import resource_plan


class DummyPSUtil:
    def __init__(self, total):
        self._total = total

    class vmem:
        def __init__(self, total):
            self.total = total

    def virtual_memory(self):
        return self.vmem(self._total)


@pytest.fixture
def fake_psutil():
    return DummyPSUtil(total=int(8e9))


def test_get_resource_plan_success(monkeypatch, fake_psutil):
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    monkeypatch.setattr(os, 'cpu_count', lambda: 4)
    plan = resource_plan.get_resource_plan()
    assert plan == {'cpu_count': 4, 'total_memory_gb': 8.0}
    monkeypatch.delitem(sys.modules, 'psutil')


def test_get_resource_plan_psutil_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'psutil':
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.setattr(os, 'cpu_count', lambda: 2)
    with caplog.at_level(logging.DEBUG, logger="src.utils.resource_plan"):
        plan = resource_plan.get_resource_plan()
    monkeypatch.setattr(builtins, '__import__', real_import)
    assert plan['total_memory_gb'] == 0.0
    assert plan['cpu_count'] == 2


def test_save_resource_plan(tmp_path, monkeypatch, fake_psutil, caplog):
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    monkeypatch.setattr(os, 'cpu_count', lambda: 1)
    with caplog.at_level(logging.INFO, logger="src.utils.resource_plan"):
        resource_plan.save_resource_plan(str(tmp_path))
    saved = json.loads((tmp_path / 'resource_plan.json').read_text())
    assert saved == {'cpu_count': 1, 'total_memory_gb': 8.0}
    monkeypatch.delitem(sys.modules, 'psutil')
