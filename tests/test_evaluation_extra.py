import json
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from joblib import dump

from src.evaluation import (
    evaluate_meta_classifier,
    walk_forward_yearly_validation,
    detect_overfit_wfv,
)


class BadModel:
    def predict_proba(self, X):
        raise RuntimeError("boom")


def _make_common_files(tmp_path, features=None, target=True):
    if features is None:
        features = ['f1']
    val = pd.DataFrame({f: np.ones(3) for f in features})
    if target:
        val['target'] = [0, 1, 0]
    val_path = tmp_path / 'val.csv'
    val.to_csv(val_path, index=False)
    feat_path = tmp_path / 'features.json'
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(features, f)
    return val_path, feat_path


def test_evaluate_meta_classifier_missing_files(tmp_path):
    val_path, feat_path = _make_common_files(tmp_path)
    res = evaluate_meta_classifier(str(tmp_path / 'no.pkl'), str(val_path), str(feat_path))
    assert res is None
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    mpath = tmp_path / 'model.pkl'
    dump(model, mpath)
    res = evaluate_meta_classifier(str(mpath), str(tmp_path / 'no.csv'), str(feat_path))
    assert res is None


def test_evaluate_meta_classifier_feature_errors(tmp_path):
    val_path, feat_path = _make_common_files(tmp_path)
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    mpath = tmp_path / 'model.pkl'
    dump(model, mpath)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(tmp_path / 'no.json'))
    assert res is None
    bad_feat = tmp_path / 'bad.json'
    with open(bad_feat, 'w', encoding='utf-8') as f:
        json.dump({'a': 1}, f)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(bad_feat))
    assert res is None
    invalid_feat = tmp_path / 'invalid.json'
    invalid_feat.write_text('{broken')
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(invalid_feat))
    assert res is None

def test_evaluate_meta_classifier_csv_and_model_errors(tmp_path):
    feat = ['f1']
    val = pd.DataFrame({'foo': [1], 'target': [1]})
    val_path = tmp_path / 'val.csv'
    val.to_csv(val_path, index=False)
    feat_path = tmp_path / 'feat.json'
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(feat, f)
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    mpath = tmp_path / 'model.pkl'
    dump(model, mpath)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None
    val = pd.DataFrame({'f1': [0, 1]})
    val.to_csv(val_path, index=False)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None
    val = pd.DataFrame({'f1': [0, 1], 'target': [0, 1]})
    feat_path.write_text(json.dumps(['f1', 'f2']))
    val.to_csv(val_path, index=False)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None
    mpath.write_text('x')
    val = pd.DataFrame({'f1': [0, 1], 'target': [0, 1]})
    val.to_csv(val_path, index=False)
    feat_path.write_text(json.dumps(['f1']))
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None
    dump(BadModel(), mpath)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None

def test_walk_forward_yearly_validation_errors(tmp_path):
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        walk_forward_yearly_validation(df, lambda d: {})
    df = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
    with pytest.raises(ValueError):
        walk_forward_yearly_validation(df, lambda d: {})


def test_walk_forward_yearly_validation_sort_and_skip():
    dates = pd.to_datetime(['2022-01-01', '2021-01-01', '2023-01-01'])
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=dates)
    res = walk_forward_yearly_validation(
        df,
        lambda d: {'pnl': d['Close'].mean(), 'winrate': 0.5, 'maxdd': 0.1},
        train_years=1,
        test_years=1,
    )
    assert list(res['fold']) == [1, 2]
    df2 = df.loc[['2022-01-01', '2021-01-01']]
    res2 = walk_forward_yearly_validation(df2.sort_index(), lambda d: {}, train_years=2, test_years=1)
    assert res2.empty


def test_detect_overfit_wfv_branches():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        detect_overfit_wfv(df)
    df = pd.DataFrame({'train_pnl': [0], 'test_pnl': [1]})
    assert not detect_overfit_wfv(df)

def test_load_json_exception(tmp_path, monkeypatch):
    val_path, feat_path = _make_common_files(tmp_path)
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    mpath = tmp_path / 'model.pkl'
    dump(model, mpath)
    def boom(path):
        raise RuntimeError('err')
    monkeypatch.setattr('src.evaluation.load_json_with_comments', boom)
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None


def test_read_csv_exception(tmp_path, monkeypatch):
    val_path, feat_path = _make_common_files(tmp_path)
    model = LogisticRegression().fit([[0], [1]], [0, 1])
    mpath = tmp_path / 'model.pkl'
    dump(model, mpath)
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('bad')))
    res = evaluate_meta_classifier(str(mpath), str(val_path), str(feat_path))
    assert res is None


def test_walk_forward_yearly_validation_continue():
    # Missing data for 2022 so second fold skipped
    dates = pd.to_datetime(['2020-01-01', '2021-01-01', '2023-01-01'])
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=dates)
    res = walk_forward_yearly_validation(
        df,
        lambda d: {'pnl': d['Close'].mean(), 'winrate': 0.5, 'maxdd': 0.1},
        train_years=1,
        test_years=1,
    )
    assert len(res) == 1

