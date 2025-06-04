import os
import logging

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
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logging.warning(f"(Warning) Environment variable {key} is not a valid float: {value}")
        return default
