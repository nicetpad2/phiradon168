from src.config_manager import ConfigManager


def test_config_manager_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv('BASE_DIR', str(tmp_path))
    cfg = ConfigManager.load()
    assert cfg.data_dir.exists()
    assert cfg.model_dir.exists()
    assert cfg.log_dir.exists()
    assert cfg.data_file_m1.suffix == '.parquet'
