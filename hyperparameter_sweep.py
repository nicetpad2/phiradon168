# -*- coding: utf-8 -*-
"""
[Patch v5.1.0] Simple hyperparameter sweep runner
สร้างสคริปต์ตัวอย่างสำหรับรันทดสอบ hyperparameter sweep
"""

"""Example script for running hyperparameter sweeps."""

from src.config import logger
import sys
from src.strategy import run_hyperparameter_sweep


def main() -> None:
    """ตัวอย่างการรัน hyperparameter sweep แบบย่อ"""

    base_params = {"output_dir": "sweep_results"}
    grid = {
        "learning_rate": [0.01, 0.05],
        "depth": [6, 8],
    }

    def dummy_train_func(**kwargs):
        # [Patch v5.1.0] ใช้ฟังก์ชันฝึกแบบจำลองจำลองเพื่อให้รันเร็ว
        print(f"เริ่มฝึก dummy_train_func ด้วยพารามิเตอร์: {kwargs}")
        return {"model": "path"}, ["feature1", "feature2"]

    results = run_hyperparameter_sweep(base_params, grid, train_func=dummy_train_func)

    for idx, res in enumerate(results, start=1):
        print(f"Run {idx}: {res}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
