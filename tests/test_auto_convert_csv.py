import pandas as pd
from src.data_loader import auto_convert_gold_csv, auto_convert_csv_to_parquet


def test_auto_convert_gold_csv_success(tmp_path):
    df = pd.DataFrame({
        'Date': ['2024-01-01'],
        'Time': ['00:00:00'],
        'open': [1.0],
        'high': [1.1],
        'low': [0.9],
        'close': [1.0],
    })
    csv = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv, index=False)
    out_f = tmp_path / 'XAUUSD_M1_thai.csv'
    auto_convert_gold_csv(str(tmp_path), output_path=str(out_f))
    assert out_f.exists()
    out = pd.read_csv(out_f, dtype=str)
    assert list(out.columns) == ['Date', 'Timestamp', 'Open', 'High', 'Low', 'Close']
    assert out.iloc[0]['Date'].startswith('2567')
    assert out.iloc[0]['Timestamp'] == '00:00:00'


def test_auto_convert_gold_csv_missing_columns(tmp_path):
    csv = tmp_path / 'XAUUSD_M15.csv'
    pd.DataFrame({'A': [1]}).to_csv(csv, index=False)
    auto_convert_gold_csv(str(tmp_path), output_path=str(tmp_path / 'XAUUSD_M15_thai.csv'))
    assert not (tmp_path / 'XAUUSD_M15_thai.csv').exists()


def test_auto_convert_gold_csv_batch(tmp_path):
    df = pd.DataFrame({
        'Date': ['2024-01-01'],
        'Time': ['00:00:00'],
        'open': [1.0],
        'high': [1.1],
        'low': [0.9],
        'close': [1.0],
    })
    for name in ['XAUUSD_M1.csv', 'XAUUSD_M15.csv']:
        df.to_csv(tmp_path / name, index=False)
    auto_convert_gold_csv(str(tmp_path), output_path=str(tmp_path))
    assert (tmp_path / 'XAUUSD_M1_thai.csv').exists()
    assert (tmp_path / 'XAUUSD_M15_thai.csv').exists()


def test_auto_convert_gold_csv_invalid_date(tmp_path):
    df = pd.DataFrame({
        'Date': ['bad'],
        'Time': ['00:00:00'],
        'Open': [1.0],
        'High': [1.1],
        'Low': [0.9],
        'Close': [1.0],
    })
    csv = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv, index=False)
    out_f = tmp_path / 'XAUUSD_M1_thai.csv'
    auto_convert_gold_csv(str(tmp_path), output_path=str(out_f))
    out = pd.read_csv(out_f)
    assert out.empty


def test_auto_convert_gold_csv_empty_dir(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'Date': ['2024-01-01'],
        'Time': ['00:00:00'],
        'Open': [1.0],
        'High': [1.1],
        'Low': [0.9],
        'Close': [1.0],
    })
    csv = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    auto_convert_gold_csv('', output_path='XAUUSD_M1.csv')
    assert (tmp_path / 'XAUUSD_M1_thai.csv').exists()


def test_auto_convert_gold_csv_timestamp_only(tmp_path):
    df = pd.DataFrame({
        'Timestamp': ['2024.01.01 00:00:00'],
        'Open': [1.0],
        'High': [1.1],
        'Low': [0.9],
        'Close': [1.0],
    })
    csv = tmp_path / 'XAUUSD_M15.csv'
    df.to_csv(csv, index=False)
    out_f = tmp_path / 'XAUUSD_M15_thai.csv'
    auto_convert_gold_csv(str(tmp_path), output_path=str(out_f))
    assert out_f.exists()
    out = pd.read_csv(out_f, dtype=str)
    assert out.iloc[0]['Timestamp'] == '00:00:00'
    assert out.iloc[0]['Date'].startswith('2567')


def test_auto_convert_gold_csv_datetime_alias(tmp_path):
    df = pd.DataFrame({
        'DateTime': ['2024-01-01 00:00:00'],
        'Open': [1.0],
        'High': [1.1],
        'Low': [0.9],
        'Close': [1.0],
    })
    csv = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv, index=False)
    out_f = tmp_path / 'XAUUSD_M1_thai.csv'
    auto_convert_gold_csv(str(tmp_path), output_path=str(out_f))
    assert out_f.exists()
    out = pd.read_csv(out_f, dtype=str)
    assert out.iloc[0]['Timestamp'] == '00:00:00'
    assert out.iloc[0]['Date'].startswith('2567')


def test_auto_convert_gold_csv_bom_header(tmp_path):
    bom_col = '\ufeffTimestamp'
    df = pd.DataFrame({bom_col: ['2024-01-01 00:00:00'], 'Open': [1.0], 'High': [1.1], 'Low': [0.9], 'Close': [1.0]})
    csv = tmp_path / 'XAUUSD_M1.csv'
    df.to_csv(csv, index=False)
    out_f = tmp_path / 'XAUUSD_M1_thai.csv'
    auto_convert_gold_csv(str(tmp_path), output_path=str(out_f))
    assert out_f.exists()
    out = pd.read_csv(out_f, dtype=str)
    assert out.iloc[0]['Timestamp'] == '00:00:00'


def test_auto_convert_csv_to_parquet_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1], 'b': [2]})
    csv = tmp_path / 'sample.csv'
    df.to_csv(csv, index=False)
    dest = tmp_path / 'out'
    dest.mkdir()
    auto_convert_csv_to_parquet(str(csv), dest)
    assert (dest / 'sample.parquet').exists() or (dest / 'sample.csv').exists()


def test_auto_convert_csv_to_parquet_missing_source(tmp_path):
    dest = tmp_path / 'out'
    dest.mkdir()
    auto_convert_csv_to_parquet(str(tmp_path / 'missing.csv'), dest)
    assert not (dest / 'missing.parquet').exists() and not (dest / 'missing.csv').exists()
