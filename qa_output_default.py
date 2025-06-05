#!/usr/bin/env python3
"""[Patch v5.7.3] QA script to list default output CSV/GZ files."""
import os
from glob import glob


def main(output_dir=".", report_file="qa_output_default_report.txt"):
    paths = glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)
    paths += glob(os.path.join(output_dir, "**", "*.gz"), recursive=True)
    lines = []
    for p in sorted(paths):
        try:
            size = os.path.getsize(p)
            lines.append(f"{p},{size}")
        except OSError as e:
            lines.append(f"{p},ERROR:{e}")
    with open(report_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"QA Report saved to {report_file} with {len(lines)} files.")
    return lines


if __name__ == "__main__":
    main()
