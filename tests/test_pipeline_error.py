import subprocess
import pytest
import main
from src.utils.errors import PipelineError


def test_run_preprocess_error(monkeypatch):
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'cmd')
    monkeypatch.setattr(subprocess, 'run', raise_error)
    with pytest.raises(PipelineError):
        main.run_preprocess(main.PipelineConfig(), runner=subprocess.run)
