import sys
import os
import pandas as pd
import pytest

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
    assert get_session_tag(ts) == 'London/New York Overlap'


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


def test_get_session_tag_end_boundary_ny():
    ts = pd.Timestamp('2024-01-01 21:00', tz='UTC')
    assert get_session_tag(ts) == 'NY'


def test_get_session_tag_end_boundary_asia():
    ts = pd.Timestamp('2024-01-01 08:00', tz='UTC')
    assert get_session_tag(ts) == 'Asia/London'


def test_get_session_tag_warn_once(caplog):
    ts1 = pd.Timestamp('2024-01-01 03:00', tz='UTC')
    ts2 = ts1 + pd.Timedelta(minutes=15)
    custom = {'Test': (0, 1)}
    with caplog.at_level('WARNING'):
        assert get_session_tag(ts1, session_times_utc=custom, warn_once=True) == 'N/A'
        assert get_session_tag(ts2, session_times_utc=custom, warn_once=True) == 'N/A'
    warnings = [r for r in caplog.records if 'out of all session ranges' in r.getMessage()]
    assert warnings


def test_get_session_tag_missing_global(monkeypatch, caplog):
    from src import utils
    # ลบตัวแปร SESSION_TIMES_UTC ชั่วคราวเพื่อทดสอบ path fallback
    backup = utils.sessions.SESSION_TIMES_UTC
    monkeypatch.delattr(utils.sessions, 'SESSION_TIMES_UTC', raising=False)
    ts = pd.Timestamp('2024-01-01 01:00', tz='UTC')
    with caplog.at_level('WARNING'):
        tag = get_session_tag(ts)
    assert tag == 'Asia'
    # คืนค่าเดิมให้ environment ทดสอบอื่น ๆ ไม่กระทบ
    monkeypatch.setattr(utils.sessions, 'SESSION_TIMES_UTC', backup, raising=False)


@pytest.mark.parametrize('hour', [23, 1])
def test_get_session_tag_tz_map_wraparound(hour):
    tz_map = {'Night': ('UTC', 22, 2)}
    ts = pd.Timestamp(f'2024-01-01 {hour:02d}:00', tz='UTC')
    assert get_session_tag(ts, session_tz_map=tz_map) == 'Night'
