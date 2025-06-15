from pathlib import Path
import pandas as pd
from scripts.convert_project_csvs import convert_project_csvs


def test_convert_project_csvs(tmp_path):
    df = pd.DataFrame({"a": [1], "b": [2]})
    base = tmp_path / "data"
    base.mkdir()
    for name in ["XAUUSD_M1.csv", "XAUUSD_M15.csv"]:
        df.to_csv(base / name, index=False)

    dest = tmp_path / "out"
    files = convert_project_csvs(str(base), str(dest))

    try:
        import pyarrow  # noqa: F401
        engine = True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            engine = True
        except Exception:
            engine = False

    if engine:
        assert len(files) == 2
        assert all(Path(f).exists() for f in files)
    else:
        assert files == []
