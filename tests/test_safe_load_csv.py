import pandas as pd
import src.data_loader as dl


def test_safe_load_csv_basic(tmp_path):
    df = pd.DataFrame({
        'Date': ['25670101', '25670101'],
        'Timestamp': ['00:00:00', '00:00:00'],
        'Open': [1, 1],
        'High': [2, 2],
        'Low': [0.5, 0.5],
        'Close': [1.5, 1.5],
        'Volume': [10, 10],
    })
    csv = tmp_path / 'in.csv'
    df.to_csv(csv, index=False)
    result = dl.safe_load_csv(str(csv))
    assert len(result) == 1
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result['Volume'].dtype == 'int64'
