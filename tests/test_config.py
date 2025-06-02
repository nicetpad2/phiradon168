import src.config as config


def test_paths():
    assert config.CSV_PATH_M1 == "/content/drive/MyDrive/Phiradon168/XAUUSD_M1.csv"
    assert config.CSV_PATH_M15 == "/content/drive/MyDrive/Phiradon168/XAUUSD_M15.csv"
    assert config.LOG_DIR == "/content/drive/MyDrive/Phiradon168/logs"
