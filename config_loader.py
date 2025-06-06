"""Utility for updating the runtime config."""

import importlib
import logging


def update_config_from_dict(params: dict) -> None:
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
