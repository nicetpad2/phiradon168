import pandas as pd
import pytest
from src.features import save_features_hdf5, load_features_hdf5


def test_features_hdf5_roundtrip(tmp_path):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    path = tmp_path / 'feat.h5'
    save_features_hdf5(df, str(path))
    assert path.exists()
    loaded = load_features_hdf5(str(path))
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)
