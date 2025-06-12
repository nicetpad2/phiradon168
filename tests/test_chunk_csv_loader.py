import pandas as pd
import src.data_loader as dl


def test_safe_load_csv_auto_basic(tmp_path):
    p = tmp_path / 'data.csv'
    pd.DataFrame({'A': range(10)}).to_csv(p, index=False)
    df = dl.safe_load_csv_auto(str(p))
    assert len(df) == 10
