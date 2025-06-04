import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.sessions import get_session_tag


def test_get_session_tag_timezone_aware():
    ts = pd.Timestamp('2024-01-01 01:00', tz='UTC')
    assert get_session_tag(ts) == 'Asia'


def test_get_session_tag_timezone_naive():
    ts = pd.Timestamp('2024-01-01 18:00')
    assert get_session_tag(ts) == 'NY'


def test_get_session_tag_overlap():
    ts = pd.Timestamp('2024-01-01 14:00', tz='UTC')
    assert get_session_tag(ts) == 'London/NY'


def test_get_session_tag_nat():
    assert get_session_tag(pd.NaT) == 'N/A'


def test_get_session_tag_string_and_cross_midnight():
    ts = '2024-01-01 23:00'
    custom = {'Night': (22, 2)}
    assert get_session_tag(ts, custom) == 'Night'
