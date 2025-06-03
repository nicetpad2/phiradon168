import logging
from pathlib import Path
from src import training


def test_save_model_creates_file(tmp_path, monkeypatch, caplog):
    called = {}
    def fake_dump(obj, path):
        called['path'] = path
        Path(path).write_text('data')
    monkeypatch.setattr(training, 'dump', fake_dump)
    training.logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO):
        training.save_model(object(), str(tmp_path), 'm')
    training.logger.setLevel(logging.WARNING)
    assert (tmp_path / 'm.joblib').is_file()
    assert called['path'] == str(tmp_path / 'm.joblib')
    assert any('Model saved' in m for m in caplog.messages)


def test_save_model_creates_qa_when_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        training.save_model(None, str(tmp_path), 'n')
    assert (tmp_path / 'n_qa.log').is_file()
    assert any('No model was trained' in m for m in caplog.messages)
