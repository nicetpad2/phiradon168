import logging
import pandas as pd
import pytest
import importlib
import wfv_runner


def _reload_runner_env(monkeypatch, data_dir, symbol, timeframe):
    """Reload wfv_runner with env variables set."""
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("SYMBOL", symbol)
    monkeypatch.setenv("TIMEFRAME", timeframe)
    cfg = importlib.import_module("src.config")
    importlib.reload(cfg)
    importlib.reload(wfv_runner)
    return wfv_runner


def _make_csv(path, rows=50):
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=rows, freq='min'),
        'Open': range(rows),
        'High': range(rows),
        'Low': range(rows),
        'Close': range(rows),
        'Volume': range(rows),
    })
    df.to_csv(path, index=False)


def test_run_walkforward_logs(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    csv = tmp_path / 'data.csv'
    _make_csv(csv)
    wfv_runner.run_walkforward(data_path=str(csv), nrows=20)
    assert any('walk-forward completed' in r.message for r in caplog.records)


def test_run_walkforward_return_frame(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    csv = tmp_path / 'data.csv'
    _make_csv(csv)
    result = wfv_runner.run_walkforward(data_path=str(csv), nrows=20)
    assert result.shape[0] == 5
    assert 'failed' in result.columns
    assert any('walk-forward completed' in r.message for r in caplog.records)


def test_run_walkforward_output_csv(tmp_path):
    path = tmp_path / 'out.csv'
    csv = tmp_path / 'data.csv'
    _make_csv(csv)
    res = wfv_runner.run_walkforward(data_path=str(csv), output_path=str(path), nrows=20)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == len(res)


def test_run_walkforward_resolve_relative(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    csv_path = data_dir / 'BTCUSD_M5.csv'
    _make_csv(csv_path)
    runner = _reload_runner_env(monkeypatch, data_dir, 'BTCUSD', 'M5')
    result = runner.run_walkforward(nrows=20)
    assert not result.empty


def test_run_walkforward_file_not_found(tmp_path, monkeypatch):
    runner = _reload_runner_env(monkeypatch, tmp_path, 'ETHUSD', 'M1')
    with pytest.raises(FileNotFoundError):
        runner.run_walkforward(data_path='missing.csv', nrows=5)


def test_run_walkforward_missing_close(tmp_path):
    path = tmp_path / 'data.csv'
    pd.DataFrame({'Timestamp': pd.date_range('2024-01-01', periods=3, freq='min'),
                  'Open': [1,2,3],
                  'High': [1,2,3],
                  'Low': [1,2,3],
                  'Volume': [1,2,3]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        wfv_runner.run_walkforward(data_path=str(path), nrows=3)
