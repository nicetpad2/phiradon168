import importlib
import config_loader


def test_update_config_from_dict(monkeypatch):
    cfg = importlib.import_module('src.config')
    monkeypatch.setattr(cfg, 'MIN_SIGNAL_SCORE_ENTRY', 0.3, raising=False)
    config_loader.update_config_from_dict({'MIN_SIGNAL_SCORE_ENTRY': 0.8})
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == 0.8


def test_update_config_adds_missing_attr(caplog):
    caplog.set_level('INFO')
    cfg = importlib.import_module('src.config')
    if hasattr(cfg, 'NEW_TEST_ATTR'):
        delattr(cfg, 'NEW_TEST_ATTR')
    config_loader.update_config_from_dict({'NEW_TEST_ATTR': 1})
    assert getattr(cfg, 'NEW_TEST_ATTR') == 1
    assert any('ไม่พบ attribute' in rec.message for rec in caplog.records)


def test_update_config_lowercase_key(monkeypatch):
    cfg = importlib.import_module('src.config')
    monkeypatch.setattr(cfg, 'FOO', 0, raising=False)
    config_loader.update_config_from_dict({'foo': 2})
    assert cfg.FOO == 2

