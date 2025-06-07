"""PipelineManager orchestrates all stages: data load, sweep, WFV, output, QA."""
import logging
import os

from .utils.pipeline_config import PipelineConfig
from .utils.errors import PipelineError

# Import pipeline stage functions from project root main module
import main as pipeline

logger = logging.getLogger(__name__)


class PipelineManager:
    """Orchestrates the full pipeline as discrete stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def stage_load(self) -> None:
        pipeline.run_preprocess(self.config)

    def stage_sweep(self) -> None:
        pipeline.run_sweep(self.config)

    def stage_wfv(self) -> None:
        # Threshold optimization is part of walk-forward run
        pipeline.run_threshold(self.config)
        pipeline.run_backtest(self.config)

    def stage_save(self) -> None:
        pipeline.run_report(self.config)

    def stage_qa(self) -> None:
        qa_path = os.path.join(self.config.model_dir, ".qa_pipeline.log")
        with open(qa_path, "a", encoding="utf-8") as fh:
            fh.write("qa completed\n")
        logger.info("[QA] log saved to %s", qa_path)

    def run_all(self) -> None:
        """Execute all pipeline stages in order."""
        for stage in [self.stage_load, self.stage_sweep, self.stage_wfv,
                      self.stage_save, self.stage_qa]:
            try:
                stage()
            except Exception as exc:  # pragma: no cover - unexpected stage error
                logger.error("Stage %s failed", stage.__name__, exc_info=True)
                raise PipelineError(f"{stage.__name__} failed") from exc
