import json
from scripts.validate_features import validate_features

def test_validate_features_ok(tmp_path):
    data = ["A", "B"]
    f = tmp_path / "features_main.json"
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    assert validate_features(str(f))

def test_validate_features_dup(tmp_path):
    data = ["A", "A"]
    f = tmp_path / "features_main.json"
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    assert not validate_features(str(f))
