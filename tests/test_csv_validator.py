import pandas as pd
import pytest

from src import csv_validator


def test_validate_and_convert_csv_success(tmp_path):
    df = pd.DataFrame({
        'Timestamp': ['2567-01-01 00:00:00', '2567-01-01 00:00:00'],
        'Open': [1.0, 1.0],
        'High': [2.0, 2.0],
        'Low': [0.5, 0.5],
        'Close': [1.5, 1.5],
        'Volume': [10, 10],
    })
    csv = tmp_path / 'in.csv'
    df.to_csv(csv, index=False)
    out = tmp_path / 'out.csv'
    result = csv_validator.validate_and_convert_csv(str(csv), str(out))
    assert out.exists()
    loaded = pd.read_csv(out)
    assert len(loaded) == 1
    assert 'Time' in loaded.columns
    assert isinstance(result.loc[0, 'Time'], pd.Timestamp)


def test_validate_and_convert_csv_with_date(tmp_path):
    df = pd.DataFrame({
        'Date': ['25670101', '25670101'],
        'Timestamp': ['00:00:00', '00:00:00'],
        'Open': [1.0, 1.0],
        'High': [2.0, 2.0],
        'Low': [0.5, 0.5],
        'Close': [1.5, 1.5],
        'Volume': [10, 10],
    })
    csv = tmp_path / 'in2.csv'
    df.to_csv(csv, index=False)
    result = csv_validator.validate_and_convert_csv(str(csv))
    assert 'Time' in result.columns
    assert len(result) == 1


def test_validate_and_convert_csv_missing(tmp_path):
    df = pd.DataFrame({'A': [1]})
    path = tmp_path / 'bad.csv'
    df.to_csv(path, index=False)
    with pytest.raises(KeyError):
        csv_validator.validate_and_convert_csv(str(path), required_cols=['Open'])


def test_validate_and_convert_csv_timestamp_only(tmp_path):
    df = pd.DataFrame(
        {
            "Timestamp": [
                "2563-06-12 03:00:00",
                "2563-06-12 03:01:00",
            ],
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [1.5, 1.5],
            "Volume": [10, 10],
        }
    )
    csv = tmp_path / "combo.csv"
    df.to_csv(csv, index=False)
    result = csv_validator.validate_and_convert_csv(str(csv))
    assert "Time" in result.columns
    assert result.iloc[0]["Time"] == pd.Timestamp("2020-06-12 03:00:00")
