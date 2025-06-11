import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.pipeline_manager import PipelineManager
from src.utils.pipeline_config import PipelineConfig


class DummyStage:
    def __init__(self, order, name):
        self.order = order
        self.name = name

    def __call__(self):
        self.order.append(self.name)


def test_run_all_invokes_stages(monkeypatch, tmp_path):
    cfg = PipelineConfig(model_dir=str(tmp_path))
    order = []
    pm = PipelineManager(cfg)
    monkeypatch.setattr(pm, 'prepare_data_environment', DummyStage(order, 'prep'))
    monkeypatch.setattr(pm, 'stage_load', DummyStage(order, 'load'))
    monkeypatch.setattr(pm, 'stage_sweep', DummyStage(order, 'sweep'))
    monkeypatch.setattr(pm, 'stage_wfv', DummyStage(order, 'wfv'))
    monkeypatch.setattr(pm, 'stage_save', DummyStage(order, 'save'))
    monkeypatch.setattr(pm, 'stage_qa', DummyStage(order, 'qa'))
    pm.run_all()
    assert order == ['prep', 'load', 'sweep', 'wfv', 'save', 'qa']
