import os
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def quick_qa_output(output_dir: str = "output_default", report_file: str = "qa_report.txt"):
    """Scan output files and report folds without trades or missing columns."""
    issues = []
    p = Path(output_dir)
    for f in p.glob("*.csv.gz"):
        try:
            df = pd.read_csv(f)
            missing_cols = [c for c in ["pnl", "entry_price"] if c not in df.columns]
            if df.empty:
                issues.append(f"{f.name}: No trades")
            elif missing_cols:
                issues.append(f"{f.name}: Missing columns {','.join(missing_cols)}")
        except Exception as e:
            issues.append(f"{f.name}: Error {e}")
    report_path = p / report_file
    with open(report_path, "w", encoding="utf-8") as fh:
        for line in issues:
            fh.write(line + "\n")
    logger.info("QA report written to %s", report_path)
    return issues
