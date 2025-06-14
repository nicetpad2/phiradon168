import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import ProjectP as proj


def test_execute_step_logs_and_returns(caplog):
    called = {}

    def dummy():
        called['ok'] = True
        return 123

    with caplog.at_level(logging.INFO, logger=proj.logger.name):
        result = proj._execute_step('dummy', dummy)

    assert result == 123
    assert called.get('ok') is True
    assert any('completed in' in rec.message and 'dummy' in rec.message for rec in caplog.records)
