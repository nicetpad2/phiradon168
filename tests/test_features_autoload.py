import os, sys, json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features


def test_load_features_autocreate(tmp_path):
    res = features.load_features_for_model('main', tmp_path)
    assert res == features.DEFAULT_META_CLASSIFIER_FEATURES
    file_path = tmp_path / 'features_main.json'
    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == features.DEFAULT_META_CLASSIFIER_FEATURES


def test_load_features_existing(tmp_path):
    file_path = tmp_path / 'features_main.json'
    custom = ['A', 'B']
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(custom, f)
    res = features.load_features_for_model('main', tmp_path)
    assert res == custom
