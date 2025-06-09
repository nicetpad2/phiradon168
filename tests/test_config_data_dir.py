import importlib
import shutil
import sys


def test_data_dir_and_hyperparams(monkeypatch):
    monkeypatch.delenv('SYMBOL', raising=False)
    monkeypatch.delenv('TIMEFRAME', raising=False)
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    cfg = importlib.import_module('src.config')
    try:
        assert cfg.DATA_DIR.is_dir()
        assert cfg.SYMBOL == 'XAUUSD'
        assert cfg.TIMEFRAME == 'M1'
        for attr in [
            'subsample',
            'colsample_bylevel',
            'bagging_temperature',
            'random_strength',
            'seed',
        ]:
            assert hasattr(cfg, attr)
            assert getattr(cfg, attr) is None
    finally:
        shutil.rmtree(cfg.DATA_DIR, ignore_errors=True)

