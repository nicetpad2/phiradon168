import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

import src.main_helpers as helpers
import src.model_helpers as models
import src.pipeline_helpers as pipe


def test_helper_stubs(tmp_path):
    assert helpers.parse_arguments() == {}
    df = pd.DataFrame({'A': [1]})
    assert helpers.drop_nan_rows(df).equals(df)
    conv = helpers.convert_to_float32(df)
    assert conv['A'].dtype == 'float32'
    out_file = tmp_path / 'a.csv'
    helpers.save_final_data(df, str(out_file))
    assert out_file.exists()


def test_model_helpers(tmp_path):
    path = models.ensure_main_features_file(str(tmp_path))
    assert os.path.exists(path)
    models.save_features_main_json([], str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), 'features_main_qa.log'))
    models.save_features_json([], 'x', str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), 'features_x.json'))


def test_pipeline_prepare_train(monkeypatch):
    called = {}
    import src.main as main_mod
    def fake_main(run_mode="FULL_PIPELINE", skip_prepare=False, suffix_from_prev_step=None):
        called['mode'] = run_mode
        return '_ok'
    monkeypatch.setattr(main_mod, 'main', fake_main)
    pipe.prepare_train_data()
    assert called['mode'] == 'PREPARE_TRAIN_DATA'
    pipe.train_models()
    assert called['mode'] == 'TRAIN_MODEL_ONLY'


def test_run_pipeline_stage_unknown():
    assert pipe.run_pipeline_stage('unknown') is None
