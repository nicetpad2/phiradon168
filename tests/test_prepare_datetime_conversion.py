import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.data_loader import prepare_datetime


def test_prepare_datetime_buddhist_conversion():
    df = pd.DataFrame({
        'Date': ['25630501', '20230501'],
        'Timestamp': ['00:00:00', '00:15:00']
    })
    result = prepare_datetime(df.copy(), 'M15')
    assert result.index[0] == pd.Timestamp('2020-05-01 00:00:00')
    assert result.index[1] == pd.Timestamp('2023-05-01 00:15:00')

