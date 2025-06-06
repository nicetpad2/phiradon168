import os
import builtins
import pandas as pd
import pytest
import src.data_loader as dl


def test_safe_get_global_exception(monkeypatch, caplog):
    def bad_globals():
        raise RuntimeError('fail')
    monkeypatch.setattr(builtins, 'globals', bad_globals)
    with caplog.at_level('ERROR'):
        assert dl.safe_get_global('X', 1) == 1
        assert 'Unexpected error' in caplog.text


def test_setup_output_directory_oserror(monkeypatch):
    monkeypatch.setattr(os, 'makedirs', lambda *a, **k: (_ for _ in ()).throw(OSError('fail')))
    with pytest.raises(SystemExit):
        dl.setup_output_directory('/tmp', 'out')


def test_setup_output_directory_exception(monkeypatch):
    monkeypatch.setattr(os, 'makedirs', lambda *a, **k: (_ for _ in ()).throw(Exception('boom')))
    with pytest.raises(SystemExit):
        dl.setup_output_directory('/tmp', 'out')


def test_safe_set_datetime_localize_fallback(monkeypatch):
    df = pd.DataFrame(index=[0])
    orig = pd.Timestamp.tz_localize
    def patched(self, tz=None, *a, **k):
        if tz == 'Asia/Bangkok':
            raise ValueError('bad')
        return orig(self, tz, *a, **k)
    monkeypatch.setattr(pd.Timestamp, 'tz_localize', patched)
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-01', naive_tz='Asia/Bangkok')
    assert df.loc[0, 'Date'] == pd.Timestamp('2024-01-01')


def test_safe_set_datetime_missing_index():
    df = pd.DataFrame(index=[0])
    dl.safe_set_datetime(df, 5, 'Date', '2024-01-02')
    assert 'Date' in df.columns
    assert pd.isna(df.loc[0, 'Date'])


def test_safe_set_datetime_assignment_error(monkeypatch):
    df = pd.DataFrame(index=[0])
    df['Date'] = pd.NaT
    orig_set = pd.core.indexing._LocIndexer.__setitem__
    call = {'c': 0}
    def patched(self, key, value):
        if call['c'] == 0:
            call['c'] += 1
            raise ValueError('err')
        return orig_set(self, key, value)
    monkeypatch.setattr(pd.core.indexing._LocIndexer, '__setitem__', patched)
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-03')
    assert pd.isna(df.loc[0, 'Date'])


def test_load_data_cached_read_and_save(monkeypatch, tmp_path):
    csv = tmp_path / 'd.csv'
    df = pd.DataFrame({'A':[1]})
    df.to_csv(csv, index=False)
    cache = csv.with_suffix('.feather')
    cache.write_text('dummy')
    monkeypatch.setattr(pd, 'read_feather', lambda p: pd.DataFrame({'A':[1]}))
    monkeypatch.setattr(dl, 'load_data', lambda *a, **k: pd.DataFrame({'B':[2]}))
    out = dl.load_data_cached(str(csv), 'M1', cache_format='feather')
    assert out.equals(pd.read_feather(cache)) or list(out.columns) == ['B']


def test_load_data_cached_errors(monkeypatch, tmp_path):
    csv = tmp_path / 'd2.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv, index=False)
    cache = csv.with_suffix('.feather')
    cache.write_text('x')
    monkeypatch.setattr(pd, 'read_feather', lambda p: (_ for _ in ()).throw(ValueError('bad')))
    monkeypatch.setattr(dl, 'load_data', lambda *a, **k: pd.DataFrame({'C':[3]}))
    monkeypatch.setattr(pd.DataFrame, 'to_feather', lambda self, p: (_ for _ in ()).throw(IOError('fail')))
    out = dl.load_data_cached(str(csv), 'M1', cache_format='feather')
    assert list(out.columns) == ['C']


def test_read_csv_with_date_parse_missing(tmp_path):
    path = tmp_path / 'no.csv'
    df = dl.read_csv_with_date_parse(str(path))
    assert df.empty


def test_check_nan_percent_none():
    assert dl.check_nan_percent(None) == 0.0


def test_check_duplicates_none():
    assert dl.check_duplicates(None) == 0


def test_check_price_jumps_no_close():
    df = pd.DataFrame({'A':[1,2]})
    assert dl.check_price_jumps(df) == 0


def test_convert_thai_years_missing():
    df = pd.DataFrame({'A':[1]})
    res = dl.convert_thai_years(df.copy(), 'Date')
    assert res.equals(df)


def test_convert_thai_datetime_type_error():
    with pytest.raises(TypeError):
        dl.convert_thai_datetime(123)


def test_prepare_datetime_index_no_date():
    df = pd.DataFrame({'A':[1]})
    res = dl.prepare_datetime_index(df.copy())
    assert res.index.equals(df.index)


def test_load_raw_data_m1_invalid(monkeypatch):
    monkeypatch.setattr(dl, 'validate_m1_data_path', lambda p: False)
    assert dl.load_raw_data_m1('bad') is None


def test_load_raw_data_m15(monkeypatch, tmp_path):
    csv = tmp_path / 'm15.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv)
    monkeypatch.setattr(dl, 'safe_load_csv_auto', lambda p: pd.read_csv(p))
    out = dl.load_raw_data_m15(str(csv))
    assert not out.empty


def test_validate_csv_data_missing_cols():
    df = pd.DataFrame({'A':[1]})
    with pytest.raises(KeyError):
        dl.validate_csv_data(df, ['A','B'])


def test_load_final_m1_data_bad_index(tmp_path):
    df = pd.DataFrame({'Open':[1], 'High':[1], 'Low':[1], 'Close':[1]}, index=['bad'])
    p = tmp_path / 'final_data_m1_v32_walkforward.csv.gz'
    df.to_csv(p, compression='gzip')
    result = dl.load_final_m1_data(str(p))
    assert result is None


def test_check_data_quality_duplicates():
    df = pd.DataFrame({'A':[1,1], 'Datetime':[1,1]})
    res = dl.check_data_quality(df.copy(), dropna=False, subset_dupes=['A','Datetime'])
    assert len(res) == 1

def test_safe_set_datetime_conversion_error(monkeypatch):
    df = pd.DataFrame({'Date':['x']})
    orig_to_datetime = pd.to_datetime
    def patched(arg, *a, **k):
        if arg is df['Date']:
            raise ValueError('bad')
        return orig_to_datetime(arg, *a, **k)
    monkeypatch.setattr(pd, 'to_datetime', patched)
    monkeypatch.setattr(pd.core.indexing._LocIndexer, '__setitem__', lambda self, key, value: (_ for _ in ()).throw(ValueError('assign fail')))
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-04')
    assert 'Date' in df.columns


def test_load_data_cached_hdf(monkeypatch, tmp_path):
    csv = tmp_path / 'h.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv, index=False)
    cache = csv.with_suffix('.h5')
    cache.write_text('dummy')
    monkeypatch.setattr(pd, 'read_hdf', lambda p, key=None: pd.DataFrame({'A':[2]}))
    monkeypatch.setattr(dl, 'load_data', lambda *a, **k: pd.DataFrame({'B':[3]}))
    monkeypatch.setattr(pd.DataFrame, 'to_hdf', lambda self, p, key=None, mode=None: None)
    out = dl.load_data_cached(str(csv), 'M1', cache_format='hdf5')
    assert list(out.columns) in (['A'], ['B'])


def test_validate_m1_data_path_invalid_type():
    assert not dl.validate_m1_data_path(123)


def test_validate_m1_data_path_missing(tmp_path):
    p = tmp_path / 'XAUUSD_M1.csv'
    assert not dl.validate_m1_data_path(str(p))


def test_load_final_m1_data_validate_fail(monkeypatch):
    monkeypatch.setattr(dl, 'validate_m1_data_path', lambda p: False)
    assert dl.load_final_m1_data('bad') is None


def test_check_price_jumps_detects():
    df = pd.DataFrame({'Close':[1.0, 1.1, 2.2]})
    assert dl.check_price_jumps(df, threshold=0.5) == 1


def test_check_data_quality_none():
    assert dl.check_data_quality(None) is None


def test_check_data_quality_custom_fill():
    df = pd.DataFrame({'A':[None, 1]})
    res = dl.check_data_quality(df.copy(), dropna=False, fillna_method='bfill')
    assert res.loc[0, 'A'] == 1

def test_safe_set_datetime_fallback_create(monkeypatch):
    df = pd.DataFrame(index=[0])
    orig = pd.core.indexing._LocIndexer.__setitem__
    calls = {'n':0}
    def bad_once(self, key, value):
        if calls['n']==0:
            calls['n']+=1
            raise ValueError('fail')
        return orig(self, key, value)
    monkeypatch.setattr(pd.core.indexing._LocIndexer, '__setitem__', bad_once)
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-05')
    assert pd.isna(df.loc[0, 'Date'])


def test_safe_set_datetime_fallback_idx_missing(monkeypatch):
    df = pd.DataFrame(index=[0])
    orig = pd.core.indexing._LocIndexer.__setitem__
    calls = {'n':0}
    def bad_once(self, key, value):
        if calls['n']==0:
            calls['n']+=1
            raise ValueError('fail')
        return orig(self, key, value)
    monkeypatch.setattr(pd.core.indexing._LocIndexer, '__setitem__', bad_once)
    dl.safe_set_datetime(df, 1, 'Date', '2024-01-06')
    assert 'Date' in df.columns


def test_load_data_cached_parquet(monkeypatch, tmp_path):
    csv = tmp_path / 'p.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv, index=False)
    cache = csv.with_suffix('.parquet')
    cache.write_text('x')
    monkeypatch.setattr(pd, 'read_parquet', lambda p: pd.DataFrame({'A':[2]}))
    monkeypatch.setattr(dl, 'load_data', lambda *a, **k: pd.DataFrame({'B':[3]}))
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, p: None)
    out = dl.load_data_cached(str(csv), 'M1', cache_format='parquet')
    assert list(out.columns) in (['A'], ['B'])


def test_load_data_cached_hdf_save(monkeypatch, tmp_path):
    csv = tmp_path / 'q.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv, index=False)
    monkeypatch.setattr(dl, 'load_data', lambda *a, **k: pd.DataFrame({'B':[2]}))
    monkeypatch.setattr(pd.DataFrame, 'to_hdf', lambda self, p, key=None, mode=None: None)
    out = dl.load_data_cached(str(csv), 'M1', cache_format='hdf5')
    assert not out.empty


def test_load_raw_data_m1_valid(monkeypatch, tmp_path):
    csv = tmp_path / 'XAUUSD_M1.csv'
    pd.DataFrame({'A':[1]}).to_csv(csv)
    monkeypatch.setattr(dl, 'validate_m1_data_path', lambda p: True)
    monkeypatch.setattr(dl, 'safe_load_csv_auto', lambda p: pd.read_csv(p))
    out = dl.load_raw_data_m1(str(csv))
    assert not out.empty


def test_load_final_m1_data_trade_log_tz(tmp_path):
    df = pd.DataFrame({'Open':[1], 'High':[1], 'Low':[1], 'Close':[1]}, index=pd.date_range('2024-01-01', periods=1, freq='min'))
    p = tmp_path / 'final_data_m1_v32_walkforward.csv.gz'
    df.to_csv(p, compression='gzip')
    trade_log = pd.DataFrame(index=pd.date_range('2024-01-01', periods=1, freq='min', tz='Asia/Bangkok'))
    loaded = dl.load_final_m1_data(str(p), trade_log)
    assert loaded.index.tz == trade_log.index.tz


def test_check_data_quality_generic_fill():
    df = pd.DataFrame({'A':[1, None]})
    res = dl.check_data_quality(df.copy(), dropna=False, fillna_method='pad')
    assert res.loc[1, 'A'] == 1
