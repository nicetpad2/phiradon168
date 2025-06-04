"""Session tagging utilities shared across modules."""

try:
    from src.config import logger
except Exception:  # pragma: no cover - fallback when config import fails
    import logging as logger
import pandas as pd

# [Patch v5.5.5] Define module-level default to avoid NameError
SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}


def get_session_tag(timestamp, session_times_utc=None, *, session_tz_map=None, naive_tz='UTC'):
    """Return trading session tag for a given timestamp.

    # [Patch] v5.4.4: Added session_tz_map and naive_tz for DST-aware tagging
    # [Patch] v5.4.8: Persist default SESSION_TIMES_UTC to suppress repeated warnings
    # [Patch] v5.5.5: Module-level default prevents missing global warnings

    Parameters
    ----------
    timestamp : pandas.Timestamp or datetime-like
        The timestamp to categorize. NaT returns "N/A".
    session_times_utc : dict, optional
        Mapping of session names to (start_hour, end_hour) in UTC.
        If None, uses global SESSION_TIMES_UTC when available.
    session_tz_map : dict, optional
        Mapping of session names to (timezone, start_hour, end_hour) where the
        hours are defined in the local timezone of that session. If provided,
        daylight saving time is handled automatically.
    naive_tz : str, optional
        Timezone to assume when ``timestamp`` is naive. Default is ``'UTC'``.
    """
    if session_times_utc is None:
        global SESSION_TIMES_UTC
        try:
            session_times_utc_local = SESSION_TIMES_UTC
        except NameError:
            logger.warning(
                "get_session_tag: Global SESSION_TIMES_UTC not found, using default.")
            SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
            session_times_utc_local = SESSION_TIMES_UTC
    else:
        session_times_utc_local = session_times_utc

    if pd.isna(timestamp):
        return "N/A"
    try:
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(naive_tz)
        ts_utc = timestamp.tz_convert('UTC')
        sessions = []
        if session_tz_map:
            for name, (tz_name, start, end) in session_tz_map.items():
                hour = ts_utc.tz_convert(tz_name).hour
                if start <= end:
                    if start <= hour < end:
                        sessions.append(name)
                else:
                    if hour >= start or hour < end:
                        sessions.append(name)
        else:
            hour = ts_utc.hour
            for name, (start, end) in session_times_utc_local.items():
                if start <= end:
                    if start <= hour < end:
                        sessions.append(name)
                else:
                    if hour >= start or hour < end:
                        sessions.append(name)
        if not sessions:
            logger.warning(f"Timestamp {timestamp} is out of all session ranges")
            return "N/A"
        return "/".join(sorted(sessions))
    except Exception as e:  # pragma: no cover - unexpected failures
        logger.error(f"   (Error) Error in get_session_tag for {timestamp}: {e}", exc_info=True)
        return "Error_Tagging"
