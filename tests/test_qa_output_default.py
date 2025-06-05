import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qa_output_default import main


def test_qa_script(tmp_path):
    f1 = tmp_path / 'a.csv'
    f1.write_text('x', encoding='utf-8')
    f2 = tmp_path / 'b.gz'
    f2.write_text('y', encoding='utf-8')
    report = tmp_path / 'r.txt'
    lines = main(output_dir=str(tmp_path), report_file=str(report))
    assert len(lines) == 2
    assert report.exists()
