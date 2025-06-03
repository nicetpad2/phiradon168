# -*- coding: utf-8 -*-
"""
[Patch v1.1.0] Hyperparameter sweep using real training
สคริปต์สำหรับรัน hyperparameter sweep และบันทึกผลลัพธ์
"""

import os
import sys
import argparse
from itertools import product
import pandas as pd

from src.config import logger
from src.training import real_train_func


def run_sweep(output_dir: str) -> None:
    """Run hyperparameter combinations and save summary."""
    learning_rates = [0.01, 0.05]
    depths = [6, 8]
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for run_id, (lr, depth) in enumerate(product(learning_rates, depths), start=1):
        print(f"เริ่มพารามิเตอร์ run {run_id}: {{'learning_rate': lr, 'depth': depth}}")
        result = real_train_func(output_dir=output_dir, learning_rate=lr, depth=depth)
        print(
            f"Run {run_id}: params={{'learning_rate': lr, 'depth': depth}}, model_path={result['model_path']}, metrics={result.get('metrics')}"
        )
        summary_rows.append(
            {
                'run_id': run_id,
                'learning_rate': lr,
                'depth': depth,
                'model_path': result['model_path'].get('model', ''),
                'features': ','.join(result.get('features', [])),
                **result.get('metrics', {}),
            }
        )

    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"Hyperparameter sweep summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='sweep_results')
    args = parser.parse_args()
    run_sweep(args.output_dir)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
