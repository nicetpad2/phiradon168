import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from src.utils import settings


def test_load_settings_missing_file(tmp_path):
    cfg = settings.load_settings(str(tmp_path / 'nope.yaml'))
    assert cfg == settings.Settings()


def test_load_settings_override_values(tmp_path):
    path = tmp_path / 's.yaml'
    path.write_text('cooldown_secs: 5\nkill_switch_pct: 0.99')
    cfg = settings.load_settings(str(path))
    assert cfg.cooldown_secs == 5
    assert cfg.kill_switch_pct == 0.99
    assert cfg.feature_format == settings.Settings.feature_format


def test_load_settings_partial_data(tmp_path):
    path = tmp_path / 's.yaml'
    path.write_text('cooldown_secs: 15')
    cfg = settings.load_settings(str(path))
    assert cfg.cooldown_secs == 15
    assert cfg.kill_switch_pct == settings.Settings.kill_switch_pct
    assert cfg.feature_format == settings.Settings.feature_format


def test_load_settings_empty_file(tmp_path):
    path = tmp_path / 'empty.yaml'
    path.write_text('')
    cfg = settings.load_settings(str(path))
    assert cfg == settings.Settings()


def test_load_settings_feature_format(tmp_path):
    path = tmp_path / 's.yaml'
    path.write_text('feature_format: hdf5')
    cfg = settings.load_settings(str(path))
    assert cfg.feature_format == 'hdf5'

