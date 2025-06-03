# -*- coding: utf-8 -*-
"""
[Patch v1.1.0] Hyperparameter sweep using real training
สคริปต์สำหรับรัน hyperparameter sweep และบันทึกผลลัพธ์

[Patch v5.2.5] รองรับการกำหนดช่วงพารามิเตอร์ผ่านบรรทัดคำสั่ง
- เพิ่มฟังก์ชัน `_parse_csv_list` สำหรับแปลงค่าคอมมา
- ปรับ `run_sweep` ให้รับพารามิเตอร์แบบกำหนดเอง
- QA: pytest -q passed
"""

import os
import sys
import argparse
from itertools import product
import pandas as pd
from typing import Callable, List

from src.config import logger
from src.training import real_train_func


def _parse_csv_list(text: str, cast: Callable) -> List:
    """แปลงสตริงคอมมาเป็นลิสต์พร้อมประเภทข้อมูล"""  # [Patch v5.2.5]
    return [cast(x.strip()) for x in text.split(',') if x.strip()]


def run_sweep(output_dir: str, learning_rates: List[float], depths: List[int]) -> None:
    """Run hyperparameter combinations and save summary."""  # [Patch v5.2.5]
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
    parser.add_argument('--learning_rates', default='0.01,0.05')
    parser.add_argument('--depths', default='6,8')
    args = parser.parse_args()

    learning_rates = _parse_csv_list(args.learning_rates, float)
    depths = _parse_csv_list(args.depths, int)

    run_sweep(args.output_dir, learning_rates, depths)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
