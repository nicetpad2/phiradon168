import os
import logging
import pytest
import pandas as pd

os.environ.setdefault("COMPACT_LOG", "1")
logging.getLogger().setLevel(logging.WARNING)

@pytest.fixture
def simple_m1_df():
    data = {
        'Open': [10, 10.1, 10.2, 10.3, 10.4],
        'High': [10.2, 10.3, 10.4, 10.5, 10.6],
        'Low': [9.8, 9.9, 10.0, 10.1, 10.2],
        'Close': [10.1, 10.2, 10.3, 10.4, 10.5],
        'Volume': [100, 110, 120, 130, 140],
        'Datetime': pd.date_range('2023-01-01', periods=5, freq='min')
    }
    df = pd.DataFrame(data)
    df.set_index('Datetime', inplace=True)
    return df


@pytest.fixture(scope='session')
def sample_data(tmp_path_factory):
    """DataFrame ขนาดเล็กสำหรับทดสอบ"""
    data = pd.DataFrame(
        {
            'timestamp': ['2025-06-07 00:00:00', '2025-06-07 00:01:00'],
            'symbol': ['XAUUSD', 'XAUUSD'],
            'side': ['BUY', 'SELL'],
            'price': [2000.0, 2001.0],
            'size': [0.01, 0.01],
            'order_type': ['LIMIT', 'LIMIT'],
            'status': ['FILLED', 'FILLED'],
        }
    )
    path = tmp_path_factory.mktemp('data') / 'sample_data.csv'
    data.to_csv(path, index=False)
    return data, path


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    warnings_count = len(terminalreporter.stats.get('warnings', []))
    terminalreporter.write_line(
        f"[SUMMARY] Passed:{passed} Failed:{failed} Warnings:{warnings_count}"
    )
