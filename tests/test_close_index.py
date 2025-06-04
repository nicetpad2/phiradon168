import pandas as pd
import logging
from src.strategy import _resolve_close_index

def test_resolve_close_index_uses_nearest(caplog):
    idxs = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:02:00"])
    df = pd.DataFrame(index=idxs)
    missing_idx = pd.Timestamp("2024-01-01 00:01:00")
    with caplog.at_level('WARNING'):
        result = _resolve_close_index(df, missing_idx, missing_idx)
    assert result in idxs
    assert "not in df_sim.index" in caplog.text
