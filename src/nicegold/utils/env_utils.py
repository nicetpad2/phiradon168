import os
import logging

logger = logging.getLogger(__name__)

# [Patch v5.5.4] Utility to read float from environment

def get_env_float(key: str, default: float) -> float:
    """Return environment variable ``key`` as float if possible.

    Parameters
    ----------
    key : str
        Environment variable name.
    default : float
        Value to return if variable is not set or invalid.

    Returns
    -------
    float
        Parsed float or ``default``.
    """
    if not isinstance(key, str):
        raise TypeError("key must be a string")

    raw_value = os.getenv(key)
    try:
        return float(raw_value)
    except TypeError:
        logger.info(f"{key} not set, using default {default}")
        return default
    except ValueError:
        logger.error(
            f"Environment variable {key} cannot be parsed as float: {raw_value}"
        )
        return default
