import logging
import pandas as pd
import pytest
import wfv_runner


def test_run_walkforward_logs(caplog):
    caplog.set_level(logging.INFO)
    wfv_runner.run_walkforward(nrows=20)
    assert any('walk-forward completed' in r.message for r in caplog.records)


def test_run_walkforward_return_frame(caplog):
    caplog.set_level(logging.INFO)
    result = wfv_runner.run_walkforward(nrows=20)
    assert result.shape[0] == 5
    assert 'failed' in result.columns
    assert any('walk-forward completed' in r.message for r in caplog.records)


def test_run_walkforward_output_csv(tmp_path):
    path = tmp_path / 'out.csv'
    res = wfv_runner.run_walkforward(output_path=str(path), nrows=20)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == len(res)


def test_run_walkforward_missing_close(tmp_path):
    path = tmp_path / 'data.csv'
    pd.DataFrame({'Open': [1,2,3]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        wfv_runner.run_walkforward(data_path=str(path), nrows=3)
