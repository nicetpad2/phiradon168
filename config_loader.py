import importlib


def update_config_from_dict(params: dict) -> None:
    """Update attributes in src.config from a dictionary."""
    cfg = importlib.import_module('src.config')
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
