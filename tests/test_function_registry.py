import ast
import os
import pytest

# Mapping of module paths to function names and their expected starting line numbers
FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version"),
    ("src/config.py", "_ensure_ta_installed"),
    ("src/config.py", "is_colab"),
    ("src/config.py", "print_gpu_utilization"),
    ("src/config.py", "show_system_status"),



    ("src/data_loader.py", "inspect_file_exists"),
    ("src/data_loader.py", "read_csv_with_date_parse"),
    ("src/data_loader.py", "check_nan_percent"),
    ("src/data_loader.py", "check_duplicates"),
    ("src/data_loader.py", "check_price_jumps"),
    ("src/data_loader.py", "convert_thai_years"),
    ("src/data_loader.py", "convert_thai_datetime"),
    ("src/data_loader.py", "prepare_datetime_index"),
    ("src/data_loader.py", "load_raw_data_m1"),
    ("src/data_loader.py", "load_raw_data_m15"),
    ("src/data_loader.py", "write_test_file"),

    ("src/data_loader.py", "validate_csv_data"),





    ("src/features.py", "tag_price_structure_patterns"),
    ("src/features.py", "calculate_trend_zone"),
    ("src/features.py", "create_session_column"),
    ("src/features.py", "fill_missing_feature_values"),
    ("src/features.py", "load_feature_config"),
    ("src/features.py", "calculate_ml_features"),

    ("src/main.py", "parse_arguments"),
    ("src/main.py", "setup_output_directory"),
    ("src/main.py", "load_features_from_file"),
    ("src/main.py", "drop_nan_rows"),
    ("src/main.py", "convert_to_float32"),
    ("src/main.py", "run_initial_backtest"),
    ("src/main.py", "save_final_data"),
    ("src/main.py", "run_auto_threshold_stage"),

    ("src/strategy.py", "run_backtest_simulation_v34"),
    ("src/strategy.py", "calculate_metrics"),
    ("src/strategy.py", "initialize_time_series_split"),
    ("src/strategy.py", "calculate_forced_entry_logic"),
    ("src/strategy.py", "apply_kill_switch"),
    ("src/strategy.py", "log_trade"),
    ("src/strategy.py", "aggregate_fold_results"),

    ("ProjectP.py", "custom_helper_function"),
]


@pytest.mark.parametrize("path, func_name", FUNCTIONS_INFO)
def test_function_exists(path: str, func_name: str) -> None:
    """Ensure each function exists in the specified module."""
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return
    assert False, f"{func_name} not found in {path}"  # pragma: no cover
