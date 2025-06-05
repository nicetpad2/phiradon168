import os
import sys
import pandas as pd
import gzip
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.qa_tools import quick_qa_output


def test_quick_qa_output(tmp_path):
    f_ok = tmp_path / 'fold1.csv.gz'
    df_ok = pd.DataFrame({'pnl':[1], 'entry_price':[1]})
    with gzip.open(f_ok, 'wt') as fh:
        df_ok.to_csv(fh, index=False)

    f_missing = tmp_path / 'fold2.csv.gz'
    df_missing = pd.DataFrame({'pnl':[ ]})
    with gzip.open(f_missing, 'wt') as fh:
        df_missing.to_csv(fh, index=False)

    issues = quick_qa_output(str(tmp_path), 'report.txt')
    assert any('No trades' in i or 'Missing columns' in i for i in issues)
    report_path = Path(tmp_path) / 'report.txt'
    assert report_path.exists()
