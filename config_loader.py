import importlib
import logging

__all__ = ["update_config_from_dict"]


# [Patch v5.9.8] Runtime config updater

def update_config_from_dict(params: dict):
    """Update attributes in :mod:`src.config` with values from ``params``.

    Parameters
    ----------
    params : dict
        Mapping of parameter names to values. Keys are matched to uppercase
        names in ``src.config`` when available; otherwise, new attributes are
        created.
    """
    try:
        cfg_module = importlib.import_module("src.config")
    except ImportError as exc:  # pragma: no cover - invalid import path
        raise ImportError(f"[Config Loader] ไม่สามารถ import โมดูล src.config: {exc}")

    for key, value in params.items():
        attr_name = key.upper() if hasattr(cfg_module, key.upper()) else key
        if not hasattr(cfg_module, key.upper()):
            logging.warning(
                f"[Config Loader] ไม่พบ attribute '{attr_name}' ใน config.py จะสร้างใหม่ใน runtime"
            )
        setattr(cfg_module, attr_name, value)
        logging.info(f"[Config Loader] ตั้งค่า config.{attr_name} = {value}")

    return cfg_module
