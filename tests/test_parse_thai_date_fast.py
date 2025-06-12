import pandas as pd
import src.data_loader as dl


def test_parse_thai_date_fast_basic():
    s = pd.Series(['24/1/2567', '1/12/2566'])
    result = dl._parse_thai_date_fast(s)
    assert result.iloc[0] == pd.Timestamp('2024-01-24')
    assert result.iloc[1] == pd.Timestamp('2023-12-01')


def test_parse_thai_date_fast_invalid():
    s = pd.Series(['invalid', '31/13/2567'])
    result = dl._parse_thai_date_fast(s)
    assert result.isna().all()
