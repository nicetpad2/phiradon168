import os
import sys
import importlib
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

config_loader = importlib.import_module('config_loader')
update_config_from_dict = config_loader.update_config_from_dict


def test_update_existing_attribute(caplog):
    cfg = importlib.import_module('src.config')
    original = cfg.MIN_SIGNAL_SCORE_ENTRY
    with caplog.at_level(logging.INFO):
        update_config_from_dict({'min_signal_score_entry': 0.77})
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == 0.77
    assert "config.MIN_SIGNAL_SCORE_ENTRY" in caplog.text
    update_config_from_dict({'min_signal_score_entry': original})


def test_create_new_attribute(caplog):
    cfg = importlib.import_module('src.config')
    if hasattr(cfg, 'new_param'):
        delattr(cfg, 'new_param')
    with caplog.at_level(logging.WARNING):
        update_config_from_dict({'new_param': 123})
    assert getattr(cfg, 'new_param') == 123
    assert "ไม่พบ attribute 'new_param'" in caplog.text
    delattr(cfg, 'new_param')
