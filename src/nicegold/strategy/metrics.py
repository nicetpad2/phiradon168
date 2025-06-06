from typing import Sequence


def calculate_metrics(pnls: Sequence[float]):
    """คำนวณ Metric พื้นฐานของชุดกำไรขาดทุน"""
    if not pnls:
        return {"r_multiple": 0.0, "winrate": 0.0}
    r_multiple = sum(pnls)
    winrate = sum(1 for p in pnls if p > 0) / len(pnls)
    return {"r_multiple": r_multiple, "winrate": winrate}
