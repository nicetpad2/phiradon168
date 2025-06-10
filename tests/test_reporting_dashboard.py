from reporting.dashboard import generate_dashboard


def test_generate_dashboard_stub(capsys):
    result = generate_dashboard({'a': 1}, output_filepath='out.html', extra=2)
    captured = capsys.readouterr()
    assert '[Dashboard Stub] Called' in captured.out
    assert result is None
