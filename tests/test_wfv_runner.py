import logging
import pandas as pd
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
