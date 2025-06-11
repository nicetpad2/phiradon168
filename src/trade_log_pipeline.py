import os
import json
import logging
import pandas as pd

from src.utils.errors import PipelineError

logger = logging.getLogger(__name__)


def load_or_generate_trade_log(log_path: str, min_rows: int = 10, features_path: str | None = None) -> pd.DataFrame:
    """Load trade log from ``log_path`` or regenerate via backtest."""
    try:
        df = pd.read_csv(log_path)
    except Exception:
        df = pd.DataFrame()

    min_rows = int(os.getenv("TRADE_LOG_MIN_ROWS", min_rows))
    if len(df) >= min_rows:
        logger.info("Loaded trade log with %d rows", len(df))
        return df

    logger.warning(
        "[Patch v6.7.4] trade_log has %d/%d rows – regenerating via backtest", len(df), min_rows
    )
    try:
        from backtest_engine import run_backtest_engine
    except Exception as exc:  # pragma: no cover
        logger.error("Cannot import backtest engine: %s", exc)
        raise PipelineError("backtest engine import failed") from exc

    features_df = pd.DataFrame()
    if features_path and os.path.exists(features_path):
        try:
            with open(features_path, "r", encoding="utf-8") as fh:
                cols = json.load(fh)
            features_df = pd.DataFrame(columns=cols)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed loading features: %s", exc)

    try:
        new_df = run_backtest_engine(features_df)
    except Exception as exc:
        logger.error("Run backtest failed: %s", exc, exc_info=True)
        raise PipelineError("trade log generation failed") from exc

    if new_df.empty:
        logger.error("Generated trade log is empty")
        raise PipelineError("trade log generation failed")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new_df.to_csv(log_path, index=False)
    logger.info("Generated trade log with %d rows", len(new_df))
    return new_df


__all__ = ["load_or_generate_trade_log"]
