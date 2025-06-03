import ast
import os
import pytest

FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 89),
    ("src/config.py", "_ensure_ta_installed", 134),
    ("src/config.py", "is_colab", 357),
    ("src/config.py", "print_gpu_utilization", 433),
    ("src/config.py", "show_system_status", 485),
    ("src/data_loader.py", "inspect_file_exists", 846),
    ("src/data_loader.py", "read_csv_with_date_parse", 851),
    ("src/data_loader.py", "check_nan_percent", 858),
    ("src/data_loader.py", "check_duplicates", 865),
    ("src/data_loader.py", "check_price_jumps", 872),
    ("src/data_loader.py", "convert_thai_years", 880),
    ("src/data_loader.py", "prepare_datetime_index", 887),
    ("src/data_loader.py", "load_raw_data_m1", 894),
    ("src/data_loader.py", "load_raw_data_m15", 899),
    ("src/data_loader.py", "write_test_file", 904),
    ("src/features.py", "calculate_trend_zone", 1194),
    ("src/features.py", "tag_price_structure_patterns", 234),
    ("src/features.py", "create_session_column", 1201),
    ("src/features.py", "fill_missing_feature_values", 1207),
    ("src/features.py", "load_feature_config", 1212),
    ("src/features.py", "calculate_ml_features", 1217),

    ("src/main.py", "parse_arguments", 1719),
    ("src/main.py", "setup_output_directory", 1724),
    ("src/main.py", "load_features_from_file", 1729),
    ("src/main.py", "drop_nan_rows", 1734),
    ("src/main.py", "convert_to_float32", 1739),
    ("src/main.py", "run_initial_backtest", 1744),
    ("src/main.py", "save_final_data", 1749),


    ("src/strategy.py", "run_backtest_simulation_v34", 1636),



    ("src/strategy.py", "initialize_time_series_split", 3745),
    ("src/strategy.py", "calculate_forced_entry_logic", 3748),
    ("src/strategy.py", "apply_kill_switch", 3751),
    ("src/strategy.py", "log_trade", 3754),
    ("src/strategy.py", "calculate_metrics", 2634),

    ("src/strategy.py", "aggregate_fold_results", 3757),





    ("ProjectP.py", "custom_helper_function", 7),
]

@pytest.mark.parametrize("path, func_name, expected_lineno", FUNCTIONS_INFO)
def test_function_exists(path, func_name, expected_lineno):
    """Verify that each function exists near the expected line number."""
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            assert abs(node.lineno - expected_lineno) <= 5, (
                f"Line mismatch for {func_name}: {node.lineno} (expected {expected_lineno})"
            )
            return
    assert False, f"{func_name} not found in {path}"
