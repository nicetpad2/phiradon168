"""PipelineManager orchestrates all stages: data load, sweep, WFV, output, QA."""
import logging
import os

from .utils.pipeline_config import PipelineConfig
from .utils.errors import PipelineError

from src.main import ensure_main_features_file
from src.trade_log_pipeline import load_or_generate_trade_log

logger = logging.getLogger(__name__)


class PipelineManager:
    """Orchestrates the full pipeline as discrete stages."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def prepare_data_environment(self) -> None:
        """Ensure required feature and trade log files are present."""
        output_dir = self.config.model_dir
        features_path = ensure_main_features_file(output_dir)
        trade_log_path = os.path.join(
            output_dir, "trade_log_v32_walkforward.csv.gz"
        )
        load_or_generate_trade_log(
            trade_log_path, min_rows=10, features_path=features_path
        )

    def stage_load(self) -> None:
        import main as pipeline
        pipeline.run_preprocess(self.config)

    def stage_sweep(self) -> None:
        import main as pipeline
        pipeline.run_sweep(self.config)

    def stage_wfv(self) -> None:
        # Threshold optimization is part of walk-forward run
        import main as pipeline
        pipeline.run_threshold(self.config)
        pipeline.run_backtest(self.config)

    def stage_save(self) -> None:
        import main as pipeline
        pipeline.run_report(self.config)

    def stage_qa(self) -> None:
        qa_path = os.path.join(self.config.model_dir, ".qa_pipeline.log")
        with open(qa_path, "a", encoding="utf-8") as fh:
            fh.write("qa completed\n")
        logger.info("[QA] log saved to %s", qa_path)

    def run_all(self) -> None:
        """Execute all pipeline stages in order."""
        try:
            self.prepare_data_environment()
        except Exception as exc:  # pragma: no cover - unexpected stage error
            logger.error("prepare_data_environment failed", exc_info=True)
            raise PipelineError("prepare_data_environment failed") from exc

        for stage in [self.stage_load, self.stage_sweep, self.stage_wfv,
                      self.stage_save, self.stage_qa]:
            try:
                stage()
            except Exception as exc:  # pragma: no cover - unexpected stage error
                logger.error("Stage %s failed", stage.__name__, exc_info=True)
                raise PipelineError(f"{stage.__name__} failed") from exc
