import ProjectP
from types import SimpleNamespace


def test_interactive_menu_select_full_pipeline(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt='': '1')
    assert ProjectP.interactive_menu() == 'full_pipeline'


def test_main_runs_menu_selection(monkeypatch):
    monkeypatch.setattr(ProjectP, 'interactive_menu', lambda: 'full_pipeline')
    called = {}
    monkeypatch.setattr(ProjectP, 'run_mode', lambda m: called.setdefault('mode', m))
    monkeypatch.setattr(ProjectP, 'parse_args', lambda: SimpleNamespace(mode='preprocess', auto_convert=False, all=False, menu=True))
    ProjectP.main()
    assert called['mode'] == 'full_pipeline'
