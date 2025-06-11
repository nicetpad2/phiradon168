"""Simple dashboard generator using pandas HTML export."""
from __future__ import annotations

import os
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_dashboard(results: Any, output_filepath: str) -> str:
    """Generate a basic HTML dashboard from a DataFrame or CSV path.

    Parameters
    ----------
    results : Any
        DataFrame or path to CSV file containing metrics.
    output_filepath : str
        Destination HTML filepath.
    """
    if isinstance(results, str):
        if os.path.exists(results):
            results = pd.read_csv(results)
        else:
            logger.error("Results path %s not found", results)
            results = pd.DataFrame()
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)
    html = results.to_html(index=False)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as fh:
        fh.write(f"<html><body>{html}</body></html>")
    logger.info("[Patch v6.6.6] Dashboard saved to %s", output_filepath)
    return output_filepath
