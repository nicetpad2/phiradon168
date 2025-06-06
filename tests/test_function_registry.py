import ast
import os
import pytest

FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 153),
    ("src/config.py", "_ensure_ta_installed", 198),
    ("src/config.py", "is_colab", 429),
    ("src/config.py", "print_gpu_utilization", 526),
    ("src/config.py", "show_system_status", 578),


    ("src/data_loader.py", "inspect_file_exists", 906),
    ("src/data_loader.py", "read_csv_with_date_parse", 911),
    ("src/data_loader.py", "check_nan_percent", 918),
    ("src/data_loader.py", "check_duplicates", 925),
    ("src/data_loader.py", "check_price_jumps", 932),
    ("src/data_loader.py", "convert_thai_years", 940),
    ("src/data_loader.py", "convert_thai_datetime", 949),
    ("src/data_loader.py", "prepare_datetime_index", 977),
    ("src/data_loader.py", "load_raw_data_m1", 1006),
    ("src/data_loader.py", "load_raw_data_m15", 1017),
    ("src/data_loader.py", "write_test_file", 1022),
    ("src/data_loader.py", "validate_csv_data", 1030),


    ("src/features.py", "calculate_trend_zone", 1529),
    ("src/features.py", "tag_price_structure_patterns", 473),
    ("src/features.py", "create_session_column", 1536),
    ("src/features.py", "fill_missing_feature_values", 1542),
    ("src/features.py", "load_feature_config", 1547),
    ("src/features.py", "calculate_ml_features", 1552),








# [Patch v5.5.3] Updated expected line numbers

    ("src/main.py", "parse_arguments", 1890),
    ("src/main.py", "setup_output_directory", 1895),
    ("src/main.py", "load_features_from_file", 1900),
    ("src/main.py", "drop_nan_rows", 1905),
    ("src/main.py", "convert_to_float32", 1910),
    ("src/main.py", "run_initial_backtest", 1915),
    ("src/main.py", "save_final_data", 1920),







    ("src/main.py", "parse_arguments", 1890),
    ("src/main.py", "setup_output_directory", 1895),
    ("src/main.py", "load_features_from_file", 1900),
    ("src/main.py", "drop_nan_rows", 1905),
    ("src/main.py", "convert_to_float32", 1910),
    ("src/main.py", "run_initial_backtest", 1915),
    ("src/main.py", "save_final_data", 1920),






    ("src/strategy.py", "run_backtest_simulation_v34", 1948),

    ("src/strategy.py", "initialize_time_series_split", 4621),
    ("src/strategy.py", "calculate_forced_entry_logic", 4624),
    ("src/strategy.py", "apply_kill_switch", 4627),
    ("src/strategy.py", "log_trade", 4630),

    ("src/strategy.py", "calculate_metrics", 3249),

    ("src/strategy.py", "aggregate_fold_results", 4633),


    ("src/strategy.py", "aggregate_fold_results", 4633),







    ("ProjectP.py", "custom_helper_function", 51),
]


@pytest.mark.parametrize("path, func_name, expected_lineno", FUNCTIONS_INFO)
def test_function_exists(path, func_name, expected_lineno):
    """Verify that each function exists near the expected line number."""
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            assert (
                abs(node.lineno - expected_lineno) <= 5
            ), f"Line mismatch for {func_name}: {node.lineno} (expected {expected_lineno})"
            return
    assert False, f"{func_name} not found in {path}"  # pragma: no cover - ensured by dataset
