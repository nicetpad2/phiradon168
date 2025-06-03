# -*- coding: utf-8 -*-
"""
[Patch v1.1.0] Hyperparameter sweep using real training
สคริปต์สำหรับรัน hyperparameter sweep และบันทึกผลลัพธ์

[Patch v5.2.5] รองรับการกำหนดช่วงพารามิเตอร์ผ่านบรรทัดคำสั่ง
- เพิ่มฟังก์ชัน `_parse_csv_list` สำหรับแปลงค่าคอมมา
- ปรับ `run_sweep` ให้รับพารามิเตอร์แบบกำหนดเอง
- QA: pytest -q passed

[Patch v5.2.6] ขยายสคริปต์ให้ครอบคลุมการใช้งานฟังก์ชันในโปรเจค
- เพิ่ม `_parse_grid_args` และ `run_general_sweep`
- `run_sweep` เรียกใช้ `run_hyperparameter_sweep` จากโมดูล strategy
- เพิ่มตัวเลือก `--grid` ใน CLI
- QA: pytest -q passed
"""

import os
import sys
import argparse
from itertools import product
import pandas as pd
from typing import Callable, List, Dict, Any

from src.config import logger
from src.training import real_train_func
from src.strategy import run_hyperparameter_sweep  # [Patch v5.2.6]


def _parse_csv_list(text: str, cast: Callable) -> List:
    """แปลงสตริงคอมมาเป็นลิสต์พร้อมประเภทข้อมูล"""  # [Patch v5.2.5]
    return [cast(x.strip()) for x in text.split(',') if x.strip()]


def _infer_type(text: str) -> Any:
    """พยายามแปลงสตริงเป็น int หรือ float ถ้าเป็นไปได้"""  # [Patch v5.2.6]
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_grid_args(entries: List[str]) -> Dict[str, List[Any]]:
    """แปลงรายการ key=v1,v2 เป็นพจนานุกรม"""  # [Patch v5.2.6]
    grid: Dict[str, List[Any]] = {}
    for entry in entries:
        if '=' not in entry:
            continue
        key, values = entry.split('=', 1)
        grid[key] = [_infer_type(v.strip()) for v in values.split(',') if v.strip()]
    return grid


def run_general_sweep(output_dir: str, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
    """รัน grid search โดยใช้ ``run_hyperparameter_sweep``"""  # [Patch v5.2.6]
    base_params = {"output_dir": output_dir}

    metrics_holder: List[Dict[str, Any]] = []

    def _adapter(**kwargs):
        """แปลงผลลัพธ์เป็น tuple ตามที่ฟังก์ชัน sweep คาดหวัง"""  # [Patch v5.2.6]
        out = real_train_func(**kwargs)
        metrics_holder.append(out.get("metrics", {}))
        return out.get("model_path", {}), out.get("features", [])

    results = run_hyperparameter_sweep(base_params, param_grid, train_func=_adapter)

    summary_rows = []
    for idx, res in enumerate(results, start=1):
        row = {"run_id": idx}
        row.update(res.get("params", {}))
        row["model_path"] = res.get("model_path", {}).get("model", "")
        row["features"] = ",".join(res.get("features", []))
        row.update(metrics_holder[idx - 1])
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Hyperparameter sweep summary saved to {summary_path}")
    return df


def run_sweep(output_dir: str, learning_rates: List[float], depths: List[int]) -> None:
    """Wrapper for backward compatibility"""  # [Patch v5.2.6]
    grid = {"learning_rate": learning_rates, "depth": depths}
    run_general_sweep(output_dir, grid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='sweep_results')
    parser.add_argument('--learning_rates', default='0.01,0.05')
    parser.add_argument('--depths', default='6,8')
    parser.add_argument('--grid', action='append', default=[], help='เพิ่มเติมพารามิเตอร์แบบ key=v1,v2')  # [Patch v5.2.6]
    args = parser.parse_args()

    learning_rates = _parse_csv_list(args.learning_rates, float)
    depths = _parse_csv_list(args.depths, int)

    extra_grid = _parse_grid_args(args.grid)
    param_grid = {'learning_rate': learning_rates, 'depth': depths, **extra_grid}

    run_general_sweep(args.output_dir, param_grid)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
