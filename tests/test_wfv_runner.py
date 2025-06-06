import wfv_runner


def test_run_walkforward_logs(caplog):
    caplog.set_level('INFO')
    wfv_runner.run_walkforward()
    assert any('[Patch] run_walkforward stub executed' in record.message for record in caplog.records)
