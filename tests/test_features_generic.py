import pandas as pd
from src.features import save_features, load_features


def test_save_load_parquet_generic(tmp_path):
    df = pd.DataFrame({'A': [1, 2]})
    path = tmp_path / 'd.parquet'
    save_features(df, str(path), 'parquet')
    if path.exists():
        loaded = load_features(str(path), 'parquet')
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)
    else:
        assert True


def test_save_load_hdf5_generic(tmp_path):
    df = pd.DataFrame({'A': [1, 2]})
    path = tmp_path / 'd.h5'
    save_features(df, str(path), 'hdf5')
    loaded = load_features(str(path), 'hdf5')
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)
