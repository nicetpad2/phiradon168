import pandas as pd
import ProjectP


def test_load_features_missing(tmp_path):
    path = tmp_path / "missing.json"
    assert ProjectP.load_features(str(path)) is None


def test_save_and_load_features_roundtrip(tmp_path):
    feats = ["A", "B"]
    path = tmp_path / "feat.json"
    ProjectP.save_features(feats, str(path))
    loaded = ProjectP.load_features(str(path))
    assert loaded == feats


def test_generate_all_features_basic(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "label": [0, 1]})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    feats = ProjectP.generate_all_features([str(csv)])
    assert "A" in feats and "B" in feats and "label" not in feats
