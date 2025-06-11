from reporting.dashboard import generate_dashboard
import pandas as pd


def test_generate_dashboard_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1, 2]})
    out = tmp_path / 'dash.html'
    result = generate_dashboard(df, output_filepath=str(out))
    assert out.exists()
    assert result == str(out)
