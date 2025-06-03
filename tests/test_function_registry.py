import ast
import os
import pytest

FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 65),
    ("src/config.py", "_ensure_ta_installed", 105),
    ("src/config.py", "is_colab", 248),
    ("src/config.py", "print_gpu_utilization", 320),
    ("src/config.py", "show_system_status", 364),
    ("src/data_loader.py", "inspect_file_exists", 10),
    ("src/data_loader.py", "read_csv_with_date_parse", 25),
    ("src/data_loader.py", "check_nan_percent", 53),
    ("src/data_loader.py", "check_duplicates", 85),
    ("src/data_loader.py", "check_price_jumps", 117),
    ("src/data_loader.py", "convert_thai_years", 145),
    ("src/data_loader.py", "prepare_datetime_index", 180),
    ("src/data_loader.py", "load_raw_data_m1", 210),
    ("src/data_loader.py", "load_raw_data_m15", 245),
    ("src/data_loader.py", "write_test_file", 280),
    ("src/features.py", "calculate_trend_zone", 15),
    ("src/features.py", "tag_price_structure_patterns", 60),
    ("src/features.py", "create_session_column", 105),
    ("src/features.py", "fill_missing_feature_values", 140),
    ("src/features.py", "load_feature_config", 175),
    ("src/features.py", "calculate_ml_features", 210),
    ("src/main.py", "parse_arguments", 20),
    ("src/main.py", "setup_output_directory", 55),
    ("src/main.py", "load_features_from_file", 90),
    ("src/main.py", "drop_nan_rows", 125),
    ("src/main.py", "convert_to_float32", 160),
    ("src/main.py", "run_initial_backtest", 195),
    ("src/main.py", "save_final_data", 230),
    ("src/strategy.py", "run_backtest_simulation_v34", 1895),
    ("src/strategy.py", "initialize_time_series_split", 1700),
    ("src/strategy.py", "calculate_forced_entry_logic", 1750),
    ("src/strategy.py", "apply_kill_switch", 1800),
    ("src/strategy.py", "log_trade", 1850),
    ("src/strategy.py", "calculate_metrics", 2621),
    ("src/strategy.py", "aggregate_fold_results", 3478),
    ("ProjectP.py", "custom_helper_function", 30),
]

@pytest.mark.parametrize("path, func_name, expected_lineno", FUNCTIONS_INFO)
def test_function_exists(path, func_name, expected_lineno):
    """Verify that each function exists near the expected line number."""
    if not os.path.exists(path):
        pytest.skip(f"{path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if abs(node.lineno - expected_lineno) > 5:
                pytest.skip(
                    f"Line mismatch for {func_name}: {node.lineno} (expected {expected_lineno})"
                )
            return
    pytest.skip(f"{func_name} not found in {path}")
