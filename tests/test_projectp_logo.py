import ProjectP


def test_print_logo_output(capfd):
    ProjectP.print_logo()
    out = capfd.readouterr().out
    assert "____" in out
