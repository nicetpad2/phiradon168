import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import src.data_loader as dl
import src.main as main


def test_inspect_file_exists_true(tmp_path):
    file_path = tmp_path / 'a.txt'
    file_path.write_text('x', encoding='utf-8')
    assert dl.inspect_file_exists(str(file_path)) is True


def test_inspect_file_exists_false(tmp_path):
    assert dl.inspect_file_exists(str(tmp_path / 'missing.txt')) is False


def test_read_csv_with_date_parse_valid(tmp_path):
    csv = tmp_path / 'f.csv'
    df_in = pd.DataFrame({'A': [1, 2]})
    df_in.to_csv(csv, index=False)
    df_out = dl.read_csv_with_date_parse(str(csv))
    pd.testing.assert_frame_equal(df_out, pd.read_csv(csv, parse_dates=True))


def test_check_nan_percent_basic():
    df = pd.DataFrame({'A': [1, np.nan]})
    assert dl.check_nan_percent(df) == pytest.approx(0.5)


def test_check_duplicates_subset():
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [1, 1, 3]})
    assert dl.check_duplicates(df, subset=['A', 'B']) == 1


def test_convert_thai_years_parses():
    df = pd.DataFrame({'Date': ['2024-01-01']})
    res = dl.convert_thai_years(df.copy(), 'Date')
    assert pd.api.types.is_datetime64_ns_dtype(res['Date'])


def test_convert_thai_datetime_valid():
    ts = dl.convert_thai_datetime('2567-01-01 00:00')
    assert ts.year == 2024


def test_convert_thai_datetime_invalid():
    ts = dl.convert_thai_datetime('notadate', errors='coerce')
    assert pd.isna(ts)


def test_prepare_datetime_index_sets_index():
    df = pd.DataFrame({'Date': ['2024-01-01']})
    res = dl.prepare_datetime_index(df.copy())
    assert isinstance(res.index, pd.DatetimeIndex)


def test_drop_nan_rows_drops_na():
    df = pd.DataFrame({'A': [1, np.nan]})
    res = main.drop_nan_rows(df)
    assert len(res) == 1


def test_convert_to_float32_dtype():
    df = pd.DataFrame({'A': [1]})
    res = main.convert_to_float32(df)
    assert res['A'].dtype == 'float32'


def test_write_test_file_creates(tmp_path):
    path = dl.write_test_file(str(tmp_path / 'w.txt'))
    assert os.path.exists(path)
