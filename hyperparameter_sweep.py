# -*- coding: utf-8 -*-
"""
[Patch v5.0.19] Simple hyperparameter sweep runner
สร้างสคริปต์ตัวอย่างสำหรับรันทดสอบ hyperparameter sweep
"""

import os
import sys

REPO_ROOT = os.path.dirname(__file__)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.strategy import run_hyperparameter_sweep


def main() -> None:
    """ตัวอย่างการรัน hyperparameter sweep แบบย่อ"""

    base_params = {"output_dir": "sweep_results"}
    grid = {
        "learning_rate": [0.01, 0.05],
        "depth": [6, 8],
    }

    def dummy_train_func(**kwargs):
        # [Patch v5.0.19] ใช้ฟังก์ชันฝึกแบบจำลองจำลองเพื่อให้รันเร็ว
        print(f"\u0e40\u0e23\u0e34\u0e48\u0e21\u0e1e\u0e32\u0e23\u0e32\u0e21\u0e34\u0e40\u0e15\u0e2d\u0e23\u0e4c: {kwargs}")
        return {"model": "path"}, ["feature1", "feature2"]

    results = run_hyperparameter_sweep(base_params, grid, train_func=dummy_train_func)

    for idx, res in enumerate(results, start=1):
        print(f"Run {idx}: {res}")


if __name__ == "__main__":
    main()
