from dataclasses import dataclass
import os
import yaml

# [Patch v6.8.5] Add feature_format option

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
DEFAULT_SETTINGS_FILE = os.path.join(CONFIG_DIR, 'settings.yaml')


@dataclass
class Settings:
    cooldown_secs: int = 60
    kill_switch_pct: float = 0.2
    feature_format: str = "parquet"
    min_signal_score_entry: float = 0.3
    meta_filter_threshold: float = 0.5
    meta_filter_relaxed_threshold: float = 0.45
    meta_filter_relax_blocks: int = 5


def load_settings(path: str = DEFAULT_SETTINGS_FILE) -> Settings:
    """Load settings from YAML file."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        # [Patch v6.8.5] Merge YAML values with dataclass defaults
        return Settings(**{**Settings().__dict__, **data})
    return Settings()


def log_settings(cfg: Settings, logger) -> None:
    """Log all setting values via provided logger."""
    from dataclasses import asdict

    for key, value in asdict(cfg).items():
        logger.info(f"[Settings] {key} = {value}")

