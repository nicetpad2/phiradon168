from dataclasses import dataclass
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
DEFAULT_SETTINGS_FILE = os.path.join(CONFIG_DIR, 'settings.yaml')


@dataclass
class Settings:
    cooldown_secs: int = 60
    kill_switch_pct: float = 0.2


def load_settings(path: str = DEFAULT_SETTINGS_FILE) -> Settings:
    """Load settings from YAML file."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        return Settings(**{**Settings().__dict__, **data})
    return Settings()

