"""Helper utilities for exporting trade logs with QA markers."""

try:
    from src.config import logger
except Exception:  # pragma: no cover - fallback only during missing config
    import logging
    logger = logging.getLogger(__name__)

import os
from dataclasses import dataclass, asdict
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd


@dataclass
class Order:
    """Simple order representation for logging."""

    side: str
    entry_price: float
    sl_price: float
    open_time: pd.Timestamp

    def as_dict(self) -> dict:
        """Return dictionary form for DataFrame conversion."""
        return asdict(self)


def setup_trade_logger(log_file: str, max_bytes: int = 1_000_000, backup_count: int = 5) -> logging.Logger:
    """[Patch v5.6.1] Setup rotating logger for trade logs."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    trade_logger = logging.getLogger("trade_logger")
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == handler.baseFilename for h in trade_logger.handlers):
        trade_logger.addHandler(handler)
    trade_logger.setLevel(logging.INFO)
    return trade_logger


def export_trade_log(trades, output_dir, label, fund_name=None):
    """[Patch v5.3.5] Export trade log ensuring file exists for every fold."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"trade_log_{label}.csv")
    qa_base_dir = os.path.join(output_dir, "qa_logs")
    if fund_name:
        qa_dir = os.path.join(qa_base_dir, fund_name)
    else:
        qa_dir = qa_base_dir
    os.makedirs(qa_dir, exist_ok=True)
    if trades is not None and not trades.empty:
        trades.to_csv(path, index=False)
        logger.info(f"[QA] Output file {path} saved successfully.")
        # [Patch] Save QA summary per trade log export
        with open(os.path.join(qa_dir, f"qa_summary_{label}.log"), "w", encoding="utf-8") as f:
            f.write(f"Trade Log QA: {len(trades)} trades, saved {path}\n")
    else:
        logger.warning(f"[QA-WARNING] No trades in {label}. Creating empty trade log.")
        pd.DataFrame().to_csv(path, index=False)
        qa_path = os.path.join(qa_dir, f"{label}_trade_qa.log")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write("[QA] No trade. Output file generated as EMPTY.\n")
        suggest_threshold_relaxation(qa_dir, label)

    # [Patch v5.9.2] Ensure BUY/SELL/NORMAL logs exist for QA
    try:
        from src.utils.trade_splitter import split_trade_log, has_buy_sell
        if trades is not None and not trades.empty and has_buy_sell(trades):
            split_trade_log(trades, output_dir)
        else:
            for fname in ("trade_log_BUY.csv", "trade_log_SELL.csv", "trade_log_NORMAL.csv"):
                fpath = os.path.join(output_dir, fname)
                if not os.path.exists(fpath):
                    pd.DataFrame().to_csv(fpath, index=False)
    except Exception as e:  # pragma: no cover - best effort QA safeguard
        logger.warning(f"[QA-WARNING] Failed to prepare side logs: {e}")


def suggest_threshold_relaxation(qa_dir: str, label: str) -> None:
    """[Patch v5.7.3] Log suggestion to relax ML threshold if no trades found."""
    os.makedirs(qa_dir, exist_ok=True)
    suggestion_file = os.path.join(qa_dir, f"relax_threshold_{label}.log")
    with open(suggestion_file, "w", encoding="utf-8") as f:
        f.write(
            "No trades generated. Consider relaxing ML_META_FILTER or entry\n"
        )
    logger.info(f"[QA] Threshold relaxation suggestion saved to {suggestion_file}")


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


def _ensure_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Return timezone-aware timestamp in UTC."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def log_open_order(order: Order, trade_log: logging.Logger = logger) -> None:
    """Log order opening info using dataclass attributes."""
    ts = _ensure_ts(order.open_time)
    trade_log.info(
        f"Order Opened: Side={order.side}, Entry={order.entry_price}, SL={order.sl_price}, Time={ts.isoformat()}"
    )


def log_close_order(
    order: Order,
    exit_price: float,
    reason: str,
    close_time: pd.Timestamp,
    trade_log: logging.Logger = logger,
) -> None:
    """Log order closing info with appropriate level."""
    ts_close = _ensure_ts(close_time)
    ts_entry = _ensure_ts(order.open_time)
    msg = (
        f"Order Closing: Time={ts_close.isoformat()}, Reason={reason}, ExitPrice={exit_price}, EntryTime={ts_entry.isoformat()}"
    )
    if "SL" in reason.upper() or "STOP LOSS" in reason.upper():
        trade_log.warning(msg)
    else:
        trade_log.info(msg)


def save_trade_snapshot(data: dict, output_file: str) -> None:
    """[Patch] Append trade snapshot data to CSV."""
    if not isinstance(data, dict):
        raise TypeError("data must be dict")
    df = pd.DataFrame([data])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    header = not os.path.exists(output_file)
    df.to_csv(output_file, mode="a", index=False, header=header)


# [Patch v5.7.3] Utility to print QA summary logs
def print_qa_summary(output_dir: str) -> str:
    """Print QA summary logs under ``output_dir/qa_logs``.

    Parameters
    ----------
    output_dir : str
        Directory containing ``qa_logs`` folder.

    Returns
    -------
    str
        Concatenated summary text. Empty string if none found.
    """
    qa_dir = os.path.join(output_dir, "qa_logs")
    if not os.path.isdir(qa_dir):
        msg = f"[QA-WARNING] QA summary directory not found: {qa_dir}"
        logger.warning(msg)
        logging.getLogger().warning(msg)
        return ""
    summaries = []
    for fname in os.listdir(qa_dir):
        if fname.startswith("qa_summary_") and fname.endswith(".log"):
            path = os.path.join(qa_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    summaries.append(f.read().strip())
            except Exception as e:
                msg = f"[QA-WARNING] Failed reading {path}: {e}"
                logger.error(msg, exc_info=True)
                logging.getLogger().error(msg)
    summary_text = "\n".join(summaries)
    if summary_text:
        logger.info(summary_text)
        logging.getLogger().info(summary_text)
    return summary_text


__all__ = [
    "Order",
    "setup_trade_logger",
    "export_trade_log",
    "aggregate_trade_logs",
    "log_open_order",
    "log_close_order",
    "print_qa_summary",
    "save_trade_snapshot",
]
