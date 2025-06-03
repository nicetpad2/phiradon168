import ast
import os
import pytest

FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 84),
    ("src/config.py", "_ensure_ta_installed", 134),
    ("src/config.py", "is_colab", 325),
    ("src/config.py", "print_gpu_utilization", 397),
    ("src/config.py", "show_system_status", 441),
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
    ("src/features.py", "calculate_trend_zone", 1184),
    ("src/features.py", "tag_price_structure_patterns", 223),
    ("src/features.py", "create_session_column", 1191),
    ("src/features.py", "fill_missing_feature_values", 1197),
    ("src/features.py", "load_feature_config", 1202),
    ("src/features.py", "calculate_ml_features", 1207),
    ("src/main.py", "parse_arguments", 1676),
    ("src/main.py", "setup_output_directory", 1681),
    ("src/main.py", "load_features_from_file", 1686),
    ("src/main.py", "drop_nan_rows", 1691),
    ("src/main.py", "convert_to_float32", 1696),
    ("src/main.py", "run_initial_backtest", 1701),
    ("src/main.py", "save_final_data", 1706),

    ("src/strategy.py", "run_backtest_simulation_v34", 1615),



    ("src/strategy.py", "initialize_time_series_split", 3724),
    ("src/strategy.py", "calculate_forced_entry_logic", 3727),
    ("src/strategy.py", "apply_kill_switch", 3730),
    ("src/strategy.py", "log_trade", 3733),
    ("src/strategy.py", "calculate_metrics", 2613),

    ("src/strategy.py", "aggregate_fold_results", 3736),





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
