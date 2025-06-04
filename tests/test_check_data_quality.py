import pandas as pd
import src.data_loader as dl


def test_check_data_quality_dropna_and_dupes():
    df = pd.DataFrame({
        'A': [1, None, 2],
        'Datetime': [1, 1, 2]
    })
    res = dl.check_data_quality(df.copy())
    assert len(res) == 2


def test_check_data_quality_fillna():
    df = pd.DataFrame({'A': [1, None], 'Datetime': [1, 2]})
    res = dl.check_data_quality(df.copy(), dropna=False, fillna_method='ffill')
    assert res.loc[1, 'A'] == 1
