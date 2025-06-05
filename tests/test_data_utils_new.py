import os
import sys
import pandas as pd
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.data_utils import convert_thai_datetime, prepare_csv_auto


def test_convert_thai_datetime_invalid(caplog):
    df = pd.DataFrame({'date': ['25650132']})
    with caplog.at_level(logging.ERROR):
        res = convert_thai_datetime(df.copy(), 'date')
    assert res['date'].isna().all()
    assert 'convert_thai_datetime failed' in caplog.text


def test_prepare_csv_auto_missing_timestamp(tmp_path, caplog):
    csv_path = tmp_path / 'test.csv'
    pd.DataFrame({'a': [1]}).to_csv(csv_path, index=False)
    with caplog.at_level(logging.WARNING):
        df = prepare_csv_auto(str(csv_path))
    assert 'timestamp column missing' in caplog.text
    assert 'a' in df.columns
