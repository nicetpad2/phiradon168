import argparse
import main as pipeline
import pytest

class DummyState:
    def __init__(self):
        self.saved = False
    def save_state(self):
        self.saved = True

def test_main_handles_keyboardinterrupt(monkeypatch, tmp_path):
    args = argparse.Namespace(mode="preprocess", config="cfg.yaml", log_level=None, debug=False, rows=None, profile=False, output_file="out.prof", live_loop=0)
    monkeypatch.setattr(pipeline, "parse_args", lambda _=None: args)
    monkeypatch.setattr(pipeline, "load_config", lambda p: pipeline.PipelineConfig())
    monkeypatch.setattr(pipeline, "run_preprocess", lambda cfg: (_ for _ in ()).throw(KeyboardInterrupt()))
    state = DummyState()
    monkeypatch.setattr(pipeline, "StateManager", lambda state_file_path='out': state)
    result = pipeline.main()
    assert result == 1
    assert state.saved

