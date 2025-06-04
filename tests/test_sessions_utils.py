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


def test_get_session_tag_custom_naive_tz():
    ts = pd.Timestamp('2024-01-01 10:00')
    tag = get_session_tag(ts, naive_tz='Asia/Bangkok')
    assert tag == 'Asia'


def test_get_session_tag_overlap():
    ts = pd.Timestamp('2024-01-01 14:00', tz='UTC')
    assert get_session_tag(ts) == 'London/NY'


def test_get_session_tag_nat():
    assert get_session_tag(pd.NaT) == 'N/A'


def test_get_session_tag_string_and_cross_midnight():
    ts = '2024-01-01 23:00'
    custom = {'Night': (22, 2)}
    assert get_session_tag(ts, custom) == 'Night'


def test_get_session_tag_dst_adjustment():
    ts = pd.Timestamp('2024-01-01 07:30', tz='UTC')
    tz_map = {
        'Asia': ('UTC', 0, 8),
        'London': ('Europe/London', 8, 17),
        'NY': ('America/New_York', 8, 17),
    }
    assert get_session_tag(ts, session_tz_map=tz_map) == 'Asia'
# DST aware test
