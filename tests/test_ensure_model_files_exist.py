import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.main as main


def _write_csv(path, df):
    p = str(path)
    if p.endswith('.gz'):
        df.to_csv(p, compression='gzip')
    else:
        df.to_csv(p)


def test_no_action_when_files_exist(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'meta_classifier.pkl').write_text('x')
    (out_dir / 'features_main.json').write_text('[]')
    (out_dir / 'meta_classifier_spike.pkl').write_text('x')
    (out_dir / 'features_spike.json').write_text('[]')
    (out_dir / 'meta_classifier_cluster.pkl').write_text('x')
    (out_dir / 'features_cluster.json').write_text('[]')

    called = {'train': False}
    monkeypatch.setattr(main, 'train_and_export_meta_model', lambda **k: ({'main': str(out_dir / 'meta_classifier.pkl')}, []))

    main.ensure_model_files_exist(str(out_dir), 'log', 'm1')
    assert not called['train']


def test_auto_train_when_missing(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    log_path = tmp_path / 'walk.csv'
    m1_path = tmp_path / 'm1.csv'
    df = pd.DataFrame({'entry_time': ['2024-01-01'], 'cluster': [0], 'spike_score': [0], 'model_tag': ['A']})
    _write_csv(log_path, df)
    m1_path.write_text('X')

    called = {}

    def dummy_train(model_purpose='main', **kwargs):
        called.setdefault('trained', []).append(model_purpose)
        (out_dir / f'meta_classifier_{model_purpose}.pkl').write_text('x')
        (out_dir / f'features_{model_purpose}.json').write_text('[]')
        return {model_purpose: str(out_dir / f'meta_classifier_{model_purpose}.pkl')}, []

    monkeypatch.setattr(main, 'train_and_export_meta_model', dummy_train)

    main.ensure_model_files_exist(str(out_dir), str(log_path)[:-4], str(m1_path)[:-4])
    assert 'main' in called.get('trained', [])
    assert 'spike' in called.get('trained', [])
    assert 'cluster' in called.get('trained', [])
    assert (out_dir / 'meta_classifier.pkl').exists()
    assert (out_dir / 'features_main.json').exists()
    assert (out_dir / 'meta_classifier_spike.pkl').exists()
    assert (out_dir / 'features_spike.json').exists()
    assert (out_dir / 'meta_classifier_cluster.pkl').exists()
    assert (out_dir / 'features_cluster.json').exists()


def test_placeholder_when_data_missing(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    main.ensure_model_files_exist(str(out_dir), 'log_missing', 'm1_missing')
    assert (out_dir / 'meta_classifier.pkl').exists()
    assert (out_dir / 'features_main.json').exists()
    assert (out_dir / 'meta_classifier_spike.pkl').exists()
    assert (out_dir / 'features_spike.json').exists()
    assert (out_dir / 'meta_classifier_cluster.pkl').exists()
    assert (out_dir / 'features_cluster.json').exists()


def test_download_feature_lists(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    feature_src = tmp_path / 'src_features.json'
    feature_src.write_text('[]')

    monkeypatch.setenv('URL_FEATURES_SPIKE', f'file://{feature_src}')
    monkeypatch.setenv('URL_FEATURES_CLUSTER', f'file://{feature_src}')
    monkeypatch.setattr(main, 'train_and_export_meta_model', lambda **k: ({}, []))

    main.ensure_model_files_exist(str(out_dir), 'log_missing', 'm1_missing')

    assert (out_dir / 'features_spike.json').exists()
    assert (out_dir / 'features_cluster.json').exists()


def test_autotrain_failure_creates_placeholders(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    log_path = tmp_path / 'walk.csv'
    m1_path = tmp_path / 'm1.csv'
    df = pd.DataFrame({'entry_time': ['2024-01-01'], 'cluster': [0], 'spike_score': [0], 'model_tag': ['A']})
    _write_csv(log_path, df)
    m1_path.write_text('X')

    def fail_train(**kw):
        raise RuntimeError('fail')

    monkeypatch.setattr(main, 'train_and_export_meta_model', fail_train)

    main.ensure_model_files_exist(str(out_dir), str(log_path)[:-4], str(m1_path)[:-4])

    for name in [
        'meta_classifier.pkl',
        'features_main.json',
        'meta_classifier_spike.pkl',
        'features_spike.json',
        'meta_classifier_cluster.pkl',
        'features_cluster.json',
    ]:
        assert (out_dir / name).exists()
