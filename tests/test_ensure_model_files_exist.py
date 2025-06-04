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

    def dummy_train(**kwargs):
        called['trained'] = True
        (out_dir / 'meta_classifier.pkl').write_text('x')
        (out_dir / 'features_main.json').write_text('[]')
        return {'main': str(out_dir / 'meta_classifier.pkl')}, []

    monkeypatch.setattr(main, 'train_and_export_meta_model', dummy_train)

    main.ensure_model_files_exist(str(out_dir), str(log_path)[:-4], str(m1_path)[:-4])
    assert called.get('trained')
    assert (out_dir / 'meta_classifier.pkl').exists()
    assert (out_dir / 'features_main.json').exists()


def test_placeholder_when_data_missing(tmp_path, monkeypatch):
    out_dir = tmp_path / 'out'
    main.ensure_model_files_exist(str(out_dir), 'log_missing', 'm1_missing')
    assert (out_dir / 'meta_classifier.pkl').exists()
    assert (out_dir / 'features_main.json').exists()
