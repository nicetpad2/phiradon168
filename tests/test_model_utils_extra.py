import os
import sys
import logging
import shutil
import pandas as pd
import numpy as np
import types
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.model_utils import (
    download_model_if_missing,
    download_feature_list_if_missing,
    evaluate_model,
    predict,
    validate_file,
)


@pytest.mark.parametrize(
    "func,file_name,env",
    [
        (download_model_if_missing, "model.pkl", "MODEL_URL"),
        (download_feature_list_if_missing, "features.json", "FEATURES_URL"),
    ],
)
def test_download_file(monkeypatch, tmp_path, func, file_name, env, caplog):
    dest = tmp_path / file_name
    src = tmp_path / ("remote_" + file_name)
    src.write_text("data")
    monkeypatch.setenv(env, src.as_uri())

    def fake_urlretrieve(url, filename):
        shutil.copyfile(url[len("file://") :], filename)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with caplog.at_level(logging.INFO, logger="src.utils.model_utils"):
        ok = func(str(dest), env)
    assert ok and dest.read_text() == "data"
    assert any("downloaded" in m.lower() for m in caplog.messages)


@pytest.mark.parametrize(
    "func,file_name,env",
    [
        (download_model_if_missing, "exists.pkl", "EXIST_MODEL"),
        (download_feature_list_if_missing, "exists.json", "EXIST_FEATURE"),
    ],
)
def test_download_already_exists(monkeypatch, tmp_path, func, file_name, env):
    dest = tmp_path / file_name
    dest.write_text("x")
    monkeypatch.setenv(env, "ignored")
    assert func(str(dest), env)


def test_download_no_url(monkeypatch, tmp_path, caplog):
    dest = tmp_path / "nope.bin"
    monkeypatch.delenv("NO_URL", raising=False)
    with caplog.at_level(logging.WARNING, logger="src.utils.model_utils"):
        ok = download_model_if_missing(str(dest), "NO_URL")
    assert not ok and not dest.exists()
    assert any("No URL specified" in m for m in caplog.messages)


def test_download_feature_list_no_url(monkeypatch, tmp_path, caplog):
    dest = tmp_path / "nope.json"
    monkeypatch.delenv("MISSING_URL", raising=False)
    with caplog.at_level(logging.WARNING, logger="src.utils.model_utils"):
        ok = download_feature_list_if_missing(str(dest), "MISSING_URL")
    assert not ok and not dest.exists()
    assert any("No URL specified" in m for m in caplog.messages)


def test_validate_file(tmp_path):
    f = tmp_path / "f.txt"
    assert not validate_file(str(f))
    f.write_text("x")
    assert validate_file(str(f))


@pytest.mark.parametrize("proba", [np.array([0.2, 0.2]), np.array([[0.8, 0.2], [0.8, 0.2]])])
def test_evaluate_single_class(proba):
    class Dummy:
        def predict_proba(self, X):
            return proba
    acc, auc = evaluate_model(Dummy(), pd.DataFrame({"a": [0, 1]}), [1, 1])
    assert isinstance(acc, float)
    assert np.isnan(auc)


def test_predict_no_proba(caplog):
    class Dummy:
        def predict(self, X):
            return [0]

    with caplog.at_level(logging.ERROR, logger="src.utils.model_utils"):
        res = predict(Dummy(), pd.DataFrame({"a": [1]}))
    assert res is None
    assert any("support predict_proba" in m for m in caplog.messages)
