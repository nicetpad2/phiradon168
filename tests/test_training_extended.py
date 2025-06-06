import pandas as pd
import logging
import src.training as training
import src.config as config


def test_optuna_sweep_no_module(caplog, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "optuna", None, raising=False)
    with caplog.at_level(logging.ERROR):
        params = training.optuna_sweep(
            pd.DataFrame({"a": [0, 1]}), pd.Series([0, 1]), n_trials=1, output_path=str(tmp_path / "m.pkl")
        )
    assert params == {}
    assert any("optuna not available" in m for m in caplog.messages)


def test_optuna_sweep_multi_trial(monkeypatch, tmp_path):
    class Trial:
        def suggest_int(self, name, low, high):
            return low

    class Study:
        def __init__(self):
            self.best_params = {}

        def optimize(self, func, n_trials):
            for _ in range(n_trials):
                func(Trial())
            self.best_params = {"n_estimators": 150, "max_depth": 5}

    class Optuna:
        def create_study(self, direction="maximize"):
            return Study()

    class DummyRF:
        def __init__(self, **kwargs):
            self.inner = training.LogisticRegression()

        def fit(self, X, y):
            self.inner.fit(X, y)

        def predict_proba(self, X):
            return self.inner.predict_proba(X)

    monkeypatch.setattr(training, "RandomForestClassifier", DummyRF, raising=False)
    monkeypatch.setattr(training, "evaluate_model", lambda *a, **k: (0.5, 0.5))
    monkeypatch.setattr(training, "USE_GPU_ACCELERATION", True, raising=False)
    monkeypatch.setattr(training, "save_model", lambda *a, **k: None)
    monkeypatch.setattr(config, "optuna", Optuna(), raising=False)
    params = training.optuna_sweep(
        pd.DataFrame({"a": [0, 1, 0, 1]}), pd.Series([0, 1, 0, 1]), n_trials=2, output_path=str(tmp_path / "m.pkl")
    )
    assert params == {"n_estimators": 150, "max_depth": 5}


def test_kfold_cv_model_catboost_missing(monkeypatch, caplog):
    monkeypatch.setattr(training, "CatBoostClassifier", None, raising=False)
    X = pd.DataFrame({"a": [0, 1], "b": [1, 0]})
    y = pd.Series([0, 1])
    with caplog.at_level(logging.ERROR):
        res = training.kfold_cv_model(X, y, model_type="catboost")
    assert res == {}
    assert any("catboost not available" in m for m in caplog.messages)
