import logging
import wfv_runner


def test_run_walkforward_logs(caplog):
    caplog.set_level(logging.INFO)
    wfv_runner.run_walkforward()
    assert any('[Patch] run_walkforward stub executed' in record.message for record in caplog.records)


def test_run_walkforward_return_and_log_once(caplog):
    caplog.set_level(logging.INFO)
    result = wfv_runner.run_walkforward()
    assert result is None
    msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert msgs.count('[Patch] run_walkforward stub executed') == 1
