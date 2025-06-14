import pandas as pd
import src.features as features


def test_build_feature_catalog_excludes_date_columns(tmp_path):
    df = pd.DataFrame({
        "Open": [1, 2],
        "Date": [20240101, 20240102],
        "Timestamp": [0, 1],
        "Close": [1.1, 2.2],
        "is_tp": [0, 0],
        "is_sl": [0, 0],
    })
    csv_path = tmp_path / "XAUUSD_M1.csv"
    df.to_csv(csv_path, index=False)

    feats = features.build_feature_catalog(str(tmp_path), str(tmp_path))
    assert "Date" not in feats
    assert "Timestamp" not in feats
    assert "Open" in feats and "Close" in feats


def test_build_feature_catalog_uses_all_rows(monkeypatch, tmp_path):
    df = pd.DataFrame({"Open": [1, 2], "Close": [2, 3]})
    csv_path = tmp_path / "XAUUSD_M1.csv"
    df.to_csv(csv_path, index=False)

    captured_kwargs = {}

    def fake_read_csv(path, **kwargs):
        captured_kwargs.update(kwargs)
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    features.build_feature_catalog(str(tmp_path), str(tmp_path))
    assert "nrows" not in captured_kwargs
