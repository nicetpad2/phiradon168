"""Simple dashboard generator using pandas HTML export."""

from __future__ import annotations

import os
import logging
from typing import Any

import plotly.graph_objects as go

import pandas as pd

logger = logging.getLogger(__name__)


def generate_dashboard(results: Any, output_filepath: str) -> str:
    """Generate an interactive HTML dashboard.

    Parameters
    ----------
    results : Any
        DataFrame or path to CSV file containing metrics.
    output_filepath : str
        Destination HTML filepath.
    """
    # [Patch v6.6.7] implement interactive chart generation
    if isinstance(results, str):
        if os.path.exists(results):
            from src.utils.data_utils import safe_read_csv

            results = safe_read_csv(results)
        else:
            logger.error("Results path %s not found", results)
            results = pd.DataFrame()
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    numeric_cols = results.select_dtypes(include="number").columns
    fig = go.Figure()
    if "fold" in results.columns and "test_pnl" in numeric_cols:
        fig.add_bar(x=results["fold"], y=results["test_pnl"], name="Test PnL")
        if "train_pnl" in numeric_cols:
            fig.add_bar(x=results["fold"], y=results["train_pnl"], name="Train PnL")
    elif len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig.add_scatter(x=list(range(len(results))), y=results[col], name=col)
    fig.update_layout(title="Metrics Summary", height=500, width=700)

    table_html = results.to_html(index=False)
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as fh:
        fh.write(
            f"<html><head><meta charset='utf-8'></head><body>{fig_html}<hr>{table_html}</body></html>"
        )

    logger.info("[Patch v6.6.7] Dashboard saved to %s", output_filepath)
    return output_filepath
