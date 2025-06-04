"""Helper utilities for exporting trade logs with QA markers."""

try:
    from src.config import logger
except Exception:  # pragma: no cover - fallback only during missing config
    import logging
    logger = logging.getLogger(__name__)

import os
import pandas as pd


def export_trade_log(trades, output_dir, label):
    """[Patch v5.3.5] Export trade log ensuring file exists for every fold."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"trade_log_{label}.csv")
    if trades is not None and not trades.empty:
        trades.to_csv(path, index=False)
        logger.info(f"[QA] Output file {path} saved successfully.")
        # [Patch] Save QA summary per trade log export
        with open(os.path.join(output_dir, f"qa_summary_{label}.log"), "w", encoding="utf-8") as f:
            f.write(f"Trade Log QA: {len(trades)} trades, saved {path}\n")
    else:
        logger.warning(f"[QA] No trades in {label}. Creating empty trade log.")
        pd.DataFrame().to_csv(path, index=False)
        qa_path = os.path.join(output_dir, f"{label}_trade_qa.log")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write("[QA] No trade. Output file generated as EMPTY.\n")


def aggregate_trade_logs(fold_dirs, output_file, label):
    """[Patch v5.4.4] Combine trade logs from multiple folds into one file."""
    dfs = []
    for directory in fold_dirs:
        path = os.path.join(directory, f"trade_log_{label}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)
    logger.info(
        f"[QA] Aggregated trade logs saved to {output_file} with {len(combined)} rows."
    )
    qa_path = os.path.splitext(output_file)[0] + "_qa.log"
    with open(qa_path, "w", encoding="utf-8") as f:
        f.write(
            f"Aggregated {len(combined)} rows from {len(fold_dirs)} folds into {output_file}\n"
        )
