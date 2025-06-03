"""Session tagging utilities shared across modules."""

import logging
import pandas as pd


def get_session_tag(timestamp, session_times_utc=None):
    """Return trading session tag for a given timestamp.

    Parameters
    ----------
    timestamp : pandas.Timestamp or datetime-like
        The timestamp to categorize. NaT returns "N/A".
    session_times_utc : dict, optional
        Mapping of session names to (start_hour, end_hour) in UTC.
        If None, uses global SESSION_TIMES_UTC when available.
    """
    if session_times_utc is None:
        global SESSION_TIMES_UTC
        try:
            session_times_utc_local = SESSION_TIMES_UTC
        except NameError:
            logging.warning(
                "get_session_tag: Global SESSION_TIMES_UTC not found, using default.")
            session_times_utc_local = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
    else:
        session_times_utc_local = session_times_utc

    if pd.isna(timestamp):
        return "N/A"
    try:
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        ts_utc = timestamp.tz_convert('UTC') if timestamp.tzinfo else timestamp.tz_localize('UTC')
        hour = ts_utc.hour
        sessions = []
        for name, (start, end) in session_times_utc_local.items():
            if start <= end:
                if start <= hour < end:
                    sessions.append(name)
            else:
                if hour >= start or hour < end:
                    sessions.append(name)
        return "/".join(sorted(sessions)) if sessions else "Other"
    except Exception as e:  # pragma: no cover - unexpected failures
        logging.error(f"   (Error) Error in get_session_tag for {timestamp}: {e}", exc_info=True)
        return "Error_Tagging"
