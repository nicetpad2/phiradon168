import pandas as pd
import pytest
from src import data_cleaner


def test_remove_duplicate_times():
    df = pd.DataFrame({
        'Time': [pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-01-01 00:00:00')],
        'Open': [1, 1],
        'High': [2, 2],
        'Low': [0.5, 0.5],
        'Close': [1.5, 1.5],
        'Volume': [10, 10],
    })
    cleaned = data_cleaner.remove_duplicate_times(df)
    assert len(cleaned) == 1


def test_clean_csv(tmp_path):
    df = pd.DataFrame(
        {
            'Date': ['25670101', '25670101'],
            'Timestamp': ['00:00:00', '00:00:00'],
            'Open': [1, 1],
            'High': [2, 2],
            'Low': [0.5, 0.5],
            'Close': [1.5, 1.5],
            'Volume': [10, 10],
        }
    )
    path = tmp_path / 'in.csv'
    df.to_csv(path, index=False)
    data_cleaner.clean_csv(str(path))
    cleaned = pd.read_csv(path)
    assert len(cleaned) == 1


def test_clean_csv_whitespace(tmp_path):
    df = pd.DataFrame(
        {
            "Date": ["25670101", "25670101"],
            "Timestamp": ["00:00:00", "00:00:00"],
            "Open": [1, 1],
            "High": [2, 2],
            "Low": [0.5, 0.5],
            "Close": [1.5, 1.5],
            "Volume": [10, 10],
        }
    )
    path = tmp_path / "space.csv"
    df.to_csv(path, index=False, sep="\t")
    data_cleaner.clean_csv(str(path))
    cleaned = pd.read_csv(path)
    assert len(cleaned) == 1


def test_clean_dataframe_basic():
    df = pd.DataFrame(
        {
            "Date": ["25670101", "25670101"],
            "Timestamp": ["00:00:00", "00:00:00"],
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [1.5, 1.5],
            "Volume": [10, 10],
        }
    )
    cleaned = data_cleaner.clean_dataframe(df)
    assert len(cleaned) == 1
    assert "Time" in cleaned.columns
    assert cleaned.iloc[0]["Time"] == pd.Timestamp("2024-01-01 00:00:00")


def test_handle_missing_fill_mean():
    df = pd.DataFrame(
        {
            "Time": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:01:00")],
            "Open": [1.0, None],
            "High": [2.0, None],
            "Low": [0.5, None],
            "Close": [1.5, None],
            "Volume": [10.0, None],
        }
    )
    out = data_cleaner.handle_missing_values(df.copy(), method="mean")
    assert out.isna().sum().sum() == 0


def test_validate_price_columns_missing():
    df = pd.DataFrame({"Time": [pd.Timestamp("2024-01-01")], "Open": [1]})
    with pytest.raises(ValueError):
        data_cleaner.validate_price_columns(df)

