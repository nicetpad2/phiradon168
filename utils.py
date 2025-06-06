import os
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

import pandas as pd


# [Patch v5.7.3] Utility helpers for data processing and resource planning

def print_qa_summary(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict[str, float]:
    """Print QA summary metrics and return them as a dictionary."""
    metrics = {
        "total_trades": 0,
        "winrate": 0.0,
        "avg_pnl": 0.0,
        "final_equity": 0.0,
        "max_drawdown": 0.0,
    }
    if trades is None or trades.empty:
        msg = (
            "\u26A0\uFE0F \u0E44\u0E21\u0E48\u0E21\u0E35\u0E44\u0E21\u0E49\u0E17\u0E35\u0E48\u0E16\u0E39\u0E01\u0E40\u0E17\u0E23\u0E14"
        )
        logger.warning(msg)
        logging.getLogger().warning(msg)
    else:
        metrics["total_trades"] = len(trades)
        if "pnl" in trades.columns:
            pnl_series = pd.to_numeric(trades["pnl"], errors="coerce")
            metrics["avg_pnl"] = float(pnl_series.mean() or 0.0)
            win_mask = pnl_series > 0
            metrics["winrate"] = float(win_mask.mean() if len(pnl_series) else 0.0)
    if equity is not None and not equity.empty:
        eq_series = equity.get("equity", equity.iloc[:, -1])
        eq_series = pd.to_numeric(eq_series, errors="coerce")
        metrics["final_equity"] = float(eq_series.dropna().iloc[-1])
        dd = (eq_series / eq_series.cummax() - 1).min()
        metrics["max_drawdown"] = float(dd if pd.notna(dd) else 0.0)
    logger.info("=== QA SUMMARY ===")
    logging.getLogger().info("=== QA SUMMARY ===")
    for k, v in metrics.items():
        logger.info("%s: %s", k, v)
        logging.getLogger().info("%s: %s", k, v)
    return metrics


def convert_thai_datetime(df: pd.DataFrame, date_col: str = "Date", time_col: str = "Timestamp") -> pd.DataFrame:
    """Convert Thai Buddhist era date and time columns to a ``timestamp`` column."""
    if date_col not in df.columns or time_col not in df.columns:
        return df

    def _convert(row):
        try:
            date_str = str(row[date_col])
            time_str = str(row[time_col])
            year = int(date_str[:4]) - 543
            greg = f"{year}{date_str[4:]} {time_str}"
            return pd.to_datetime(greg, format="%Y%m%d %H:%M:%S", errors="coerce")
        except Exception as e:  # pragma: no cover - unexpected formats
            with open("error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"convert_thai_datetime error: {e}\n")
            return pd.NaT

    df = df.copy()
    df["timestamp"] = df.apply(_convert, axis=1)
    return df


def prepare_csv_auto(path: str) -> pd.DataFrame:
    """Load a CSV file and convert Thai datetime columns automatically."""
    df = pd.read_csv(path)
    df = convert_thai_datetime(df)
    return df


def get_resource_plan(debug: bool = False) -> Dict[str, object]:
    """Return basic resource usage information."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024 ** 3
        threads = psutil.cpu_count(logical=True)
    except ModuleNotFoundError:
        ram_gb, threads = 0.0, 2
    except AttributeError:
        ram_gb, threads = 0.0, 2

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception as e:  # pragma: no cover - device query failed
                logging.warning("GPU detected but name unavailable: %s", e)
                gpu_name = "Unknown"
            device = "cuda"
        else:
            device = "cpu"
            gpu_name = "None"
    except ModuleNotFoundError:
        device = "cpu"
        gpu_name = "Unknown"
    except AttributeError:
        device = "cpu"
        gpu_name = "Unknown"

    plan = {
        "ram_gb": round(ram_gb, 2),
        "threads": threads,
        "device": device,
        "gpu_name": gpu_name,
    }

    if debug or os.getenv("DEBUG_RESOURCE"):
        with open("resource_debug.log", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(plan, indent=2))
    return plan


__all__ = [
    "print_qa_summary",
    "convert_thai_datetime",
    "prepare_csv_auto",
    "get_resource_plan",
]


