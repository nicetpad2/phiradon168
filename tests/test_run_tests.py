import types
import sys
import pytest
import run_tests

class DummySummary:
    def __init__(self):
        self.total = 1
        self.passed = 1
        self.failed = 0
        self.skipped = 0

def _patch_pytest(monkeypatch, called):
    def fake_main(args, plugins=None):
        called['args'] = args
        return 0
    monkeypatch.setattr(run_tests, 'pytest', types.SimpleNamespace(main=fake_main))
    monkeypatch.setattr(run_tests, '_SummaryPlugin', lambda: DummySummary())


def test_run_tests_enables_auto_parallel(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(sys, 'argv', ['run_tests.py'])
    with pytest.raises(SystemExit) as exc:
        run_tests.main()
    assert exc.value.code == 0
    assert '-n' in called['args']
    assert 'auto' in called['args']


def test_run_tests_last_failed(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(sys, 'argv', ['run_tests.py', '--lf'])
    with pytest.raises(SystemExit):
        run_tests.main()
    assert '--last-failed' in called['args']
