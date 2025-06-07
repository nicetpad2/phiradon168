from typing import Sequence


def compute_metrics(values: Sequence[float]):
    """คำนวณค่าเฉลี่ยจากลิสต์ตัวเลข"""
    if not values:
        mean = 0.0
    else:
        mean = sum(values) / len(values)
    return {"mean": mean}
