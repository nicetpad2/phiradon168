import logging
import wfv_runner


def test_run_walkforward_logs(caplog):
    caplog.set_level(logging.INFO)
    wfv_runner.run_walkforward()
    assert any('walk-forward completed' in r.message for r in caplog.records)


def test_run_walkforward_return_frame(caplog):
    caplog.set_level(logging.INFO)
    result = wfv_runner.run_walkforward()
    assert result.shape[0] == 5
    assert 'failed' in result.columns
    assert any('walk-forward completed' in r.message for r in caplog.records)
