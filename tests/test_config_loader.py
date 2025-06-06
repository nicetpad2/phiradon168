import importlib
import config_loader


def test_update_config_from_dict(monkeypatch):
    cfg = importlib.import_module('src.config')
    monkeypatch.setattr(cfg, 'MIN_SIGNAL_SCORE_ENTRY', 0.3, raising=False)
    config_loader.update_config_from_dict({'MIN_SIGNAL_SCORE_ENTRY': 0.8})
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == 0.8
