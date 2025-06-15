"""Utility for updating the runtime config."""

import importlib
import logging
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigManager:
    """Singleton จัดการโหลดและเข้าถึงไฟล์คอนฟิก YAML."""

    _instance: Optional["ConfigManager"] = None

    def __new__(cls, *args, **kwargs) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 settings_path: str = "config/settings.yaml",
                 pipeline_path: str = "config/pipeline.yaml") -> None:
        if hasattr(self, "initialized"):
            return
        self.settings_path = Path(settings_path)
        self.pipeline_path = Path(pipeline_path)
        self.settings = self._load_yaml(self.settings_path)
        self.pipeline = self._load_yaml(self.pipeline_path)
        self.initialized = True

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_setting(self, key: str, default: Any | None = None) -> Any:
        """คืนค่าจาก settings หากไม่พบจะคืน ``default``"""
        return self.settings.get(key, default)


from types import ModuleType


def update_config_from_dict(params: dict) -> ModuleType:
    """Update :mod:`src.config` attributes from ``params``.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values to write into ``src.config``.
    """
    try:
        cfg_module = importlib.import_module("src.config")
    except ImportError as exc:  # pragma: no cover - unexpected missing module
        raise ImportError(
            f"[Config Loader] ไม่สามารถ import โมดูล src.config: {exc}"
        ) from exc

    for key, value in params.items():
        # [Patch] allow lowercase keys by mapping to uppercase attributes
        attr_name = key.upper() if hasattr(cfg_module, key.upper()) else key
        if not hasattr(cfg_module, attr_name):
            logging.warning(
                f"[Config Loader] ไม่พบ attribute '{attr_name}' ใน config.py จะสร้างใหม่ใน runtime"
            )
        setattr(cfg_module, attr_name, value)
        logging.info(f"[Config Loader] ตั้งค่า config.{attr_name} = {value}")

    # return module for test convenience
    return cfg_module
