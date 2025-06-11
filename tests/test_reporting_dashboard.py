from reporting.dashboard import generate_dashboard
import pandas as pd


def test_generate_dashboard_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1, 2]})
    out = tmp_path / 'dash.html'
    result = generate_dashboard(df, output_filepath=str(out))
    assert out.exists()
    assert result == str(out)


def test_generate_dashboard_from_csv(tmp_path):
    df = pd.DataFrame({'fold': [1], 'test_pnl': [1.0]})
    csv = tmp_path / 'm.csv'
    df.to_csv(csv, index=False)
    out = tmp_path / 'dash_csv.html'
    result = generate_dashboard(str(csv), output_filepath=str(out))
    assert out.exists()
    assert result == str(out)


def test_generate_dashboard_missing_file(tmp_path, caplog):
    import logging
    caplog.set_level(logging.ERROR)
    out = tmp_path / 'missing.html'
    result = generate_dashboard('no_file.csv', output_filepath=str(out))
    assert out.exists()
    assert result == str(out)
