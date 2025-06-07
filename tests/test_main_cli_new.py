import os
import sys
import subprocess
import builtins
import io
import yaml
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import main as pipeline
from src.utils.pipeline_config import PipelineConfig


def test_setup_logging_override_level(monkeypatch):
    yaml_cfg = {'version': 1, 'root': {'level': 'INFO', 'handlers': []}}
    yaml_text = yaml.dump(yaml_cfg)

    def fake_open(file, mode='r', *args, **kwargs):
        if file == 'config/logger_config.yaml':
            return io.StringIO(yaml_text)
        return builtins.open(file, mode, *args, **kwargs)

    captured = {}

    monkeypatch.setattr(builtins, 'open', fake_open)
    monkeypatch.setattr(pipeline.logging.config, 'dictConfig', lambda cfg: captured.setdefault('level', cfg['root']['level']))

    pipeline.setup_logging('debug')

    assert captured['level'] == 'DEBUG'


def test_run_threshold_success():
    called = {}

    def fake_run(cmd, check):
        called['cmd'] = cmd
        called['check'] = check

    pipeline.run_threshold(PipelineConfig(), runner=fake_run)
    assert 'threshold_optimization.py' in called['cmd'][1]
    assert called['check']


def test_run_sweep_failure():
    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(pipeline.PipelineError):
        pipeline.run_sweep(PipelineConfig(), runner=fake_run)
