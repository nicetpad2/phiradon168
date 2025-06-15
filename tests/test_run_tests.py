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


def test_find_changed_tests(monkeypatch):
    monkeypatch.setattr(
        run_tests.subprocess,
        'check_output',
        lambda cmd, text=True: 'tests/test_a.py\nsrc/mod.py\n',
    )
    result = run_tests.find_changed_tests('HEAD~1')
    assert result == ['tests/test_a.py']


def test_run_tests_changed(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(run_tests, 'find_changed_tests', lambda base: ['tests/test_a.py'])
    monkeypatch.setattr(sys, 'argv', ['run_tests.py', '--changed'])
    with pytest.raises(SystemExit):
        run_tests.main()
    assert 'tests/test_a.py' in called['args']


def test_run_tests_cov(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(sys, 'argv', ['run_tests.py', '--cov'])
    with pytest.raises(SystemExit):
        run_tests.main()
    assert '--cov' in called['args']
    assert 'src' in called['args']


def test_run_tests_maxfail(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(sys, 'argv', ['run_tests.py', '--maxfail', '2'])
    with pytest.raises(SystemExit):
        run_tests.main()
    assert '--maxfail' in called['args']
    assert '2' in called['args']


def test_run_tests_fast_sets_maxfail(monkeypatch):
    called = {}
    _patch_pytest(monkeypatch, called)
    monkeypatch.setattr(sys, 'argv', ['run_tests.py', '--fast'])
    with pytest.raises(SystemExit):
        run_tests.main()
    assert '-m' in called['args'] and 'not integration' in called['args']
    assert '--maxfail' in called['args'] and '1' in called['args']
