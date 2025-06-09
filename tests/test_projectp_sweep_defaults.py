import ProjectP


def test_sweep_params_contains_new_keys():
    params = ProjectP.DEFAULT_SWEEP_PARAMS
    assert 'bagging_temperature' in params
    assert 'random_strength' in params
    assert 'subsample' in params
    assert 'colsample_bylevel' in params
