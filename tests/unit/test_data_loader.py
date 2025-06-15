import os
import sys
import pandas as pd
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.utils import data_utils as du
from src import data_loader as dl


def test_prepare_csv_auto_not_found(tmp_path, caplog):
    path = tmp_path / 'missing.csv'
    with caplog.at_level(logging.ERROR):
        df = du.prepare_csv_auto(str(path))
    assert df.empty
    assert 'file not found' in caplog.text


def test_convert_thai_datetime_missing_column():
    df = pd.DataFrame({'a': [1]})
    result = du.convert_thai_datetime(df, 'time')
    assert result.equals(df)


def test_setup_output_directory(tmp_path):
    base = tmp_path
    result = dl.setup_output_directory(str(base), 'out')
    assert os.path.isdir(result)


def test_extract_thai_date_time_vec():
    ts_series = pd.Series(["2024-01-01 00:00:00", "2567-01-02 01:02:03", "bad"])
    result = dl._extract_thai_date_time_vec(ts_series)
    assert result.loc[0, "Date"] == "2567" + "0101"
    assert result.loc[0, "Timestamp"] == "00:00:00"
    assert result.loc[1, "Date"] == "2567" + "0102"
    assert result.loc[1, "Timestamp"] == "01:02:03"
    assert pd.isna(result.loc[2, "Date"]) and pd.isna(result.loc[2, "Timestamp"])
