import pandas as pd
from src import data_cleaner


def test_remove_duplicates_index():
    df = pd.DataFrame({'A': [1, 2, 2]}, index=[0, 0, 1])
    cleaned = data_cleaner.remove_duplicates(df)
    assert len(cleaned) == 2
    assert not cleaned.index.has_duplicates


def test_clean_csv(tmp_path):
    df = pd.DataFrame({'Date': [1, 1], 'Timestamp': [1, 1], 'A': [1, 2]})
    path = tmp_path / 'in.csv'
    df.to_csv(path, index=False)
    data_cleaner.clean_csv(str(path))
    cleaned = pd.read_csv(path)
    assert len(cleaned) == 1
