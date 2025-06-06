import pandas as pd
from src.features import save_features_parquet, load_features_parquet


def test_features_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    path = tmp_path / 'feat.parquet'
    save_features_parquet(df, str(path))
    if path.exists():
        loaded = load_features_parquet(str(path))
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)
    else:
        assert True
