import ProjectP as proj

def test_parse_args_modes():
    assert proj.parse_args(["--mode", "sweep"]).mode == "sweep"
    assert proj.parse_args([]).mode == "preprocess"
