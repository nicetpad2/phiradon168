import os
import src.training as training


def test_real_train_func_accepts_seed(tmp_path):
    res = training.real_train_func(output_dir=str(tmp_path), seed=123)
    assert 'model_path' in res
    assert os.path.exists(res['model_path']['model'])
    assert 'metrics' in res
