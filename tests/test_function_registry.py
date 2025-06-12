import ast
import os
import pytest

# Mapping of module paths to function names and their expected starting line numbers
FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 222),
    ("src/config.py", "_ensure_ta_installed", 267),
    ("src/config.py", "is_colab", 498),
    ("src/config.py", "print_gpu_utilization", 598),
    ("src/config.py", "show_system_status", 650),


    ("src/data_loader.py", "inspect_file_exists", 1024),
    ("src/data_loader.py", "read_csv_with_date_parse", 1029),
    ("src/data_loader.py", "check_nan_percent", 1036),
    ("src/data_loader.py", "check_duplicates", 1043),
    ("src/data_loader.py", "check_price_jumps", 1050),
    ("src/data_loader.py", "convert_thai_years", 1092),
    ("src/data_loader.py", "convert_thai_datetime", 1099),
    ("src/data_loader.py", "prepare_datetime_index", 1127),
    ("src/data_loader.py", "load_raw_data_m1", 1182),
    ("src/data_loader.py", "load_raw_data_m15", 1192),
    ("src/data_loader.py", "write_test_file", 1198),
    ("src/data_loader.py", "validate_csv_data", 1300),


    ("src/features.py", "tag_price_structure_patterns", 473),
    ("src/features.py", "calculate_trend_zone", 1572),
    ("src/features.py", "create_session_column", 1583),
    ("src/features.py", "fill_missing_feature_values", 1609),
    ("src/features.py", "load_feature_config", 1614),
    ("src/features.py", "calculate_ml_features", 1619),

    ("src/main.py", "parse_arguments", 2005),
    ("src/main.py", "setup_output_directory", 2010),
    ("src/main.py", "load_features_from_file", 2015),
    ("src/main.py", "drop_nan_rows", 2020),
    ("src/main.py", "convert_to_float32", 2025),
    ("src/main.py", "run_initial_backtest", 2030),
    ("src/main.py", "save_final_data", 2035),
    ("src/main.py", "run_auto_threshold_stage", 2041),

    ("src/strategy.py", "run_backtest_simulation_v34", 1941),
    ("src/strategy.py", "calculate_metrics", 3233),
    ("src/strategy.py", "initialize_time_series_split", 4605),
    ("src/strategy.py", "calculate_forced_entry_logic", 4608),
    ("src/strategy.py", "apply_kill_switch", 4611),
    ("src/strategy.py", "log_trade", 4614),
    ("src/strategy.py", "aggregate_fold_results", 4617),

    ("ProjectP.py", "custom_helper_function", 104),
]


@pytest.mark.parametrize("path, func_name, expected_lineno", FUNCTIONS_INFO)
def test_function_exists(path: str, func_name: str, expected_lineno: int) -> None:
    """Ensure each function exists roughly at the expected line number."""
    assert os.path.exists(path), f"{path} does not exist"
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            assert abs(node.lineno - expected_lineno) <= 5, (
                f"Line mismatch for {func_name}: {node.lineno} (expected {expected_lineno})"
            )
            return
    assert False, f"{func_name} not found in {path}"  # pragma: no cover
