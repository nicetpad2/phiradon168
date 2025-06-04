import ast
import os
import pytest

FUNCTIONS_INFO = [
    ("src/config.py", "log_library_version", 88),
    ("src/config.py", "_ensure_ta_installed", 133),
    ("src/config.py", "is_colab", 362),
    ("src/config.py", "print_gpu_utilization", 458),
    ("src/config.py", "show_system_status", 510),
    ("src/data_loader.py", "inspect_file_exists", 828),
    ("src/data_loader.py", "read_csv_with_date_parse", 833),
    ("src/data_loader.py", "check_nan_percent", 840),
    ("src/data_loader.py", "check_duplicates", 847),
    ("src/data_loader.py", "check_price_jumps", 854),
    ("src/data_loader.py", "convert_thai_years", 862),
    ("src/data_loader.py", "prepare_datetime_index", 869),
    ("src/data_loader.py", "load_raw_data_m1", 898),
    ("src/data_loader.py", "load_raw_data_m15", 909),
    ("src/data_loader.py", "write_test_file", 914),
    ("src/features.py", "calculate_trend_zone", 1225),
    ("src/features.py", "tag_price_structure_patterns", 234),
    ("src/features.py", "create_session_column", 1232),
    ("src/features.py", "fill_missing_feature_values", 1238),
    ("src/features.py", "load_feature_config", 1243),
    ("src/features.py", "calculate_ml_features", 1248),






# [Patch v5.5.1] Updated expected line numbers
    ("src/main.py", "parse_arguments", 1867),
    ("src/main.py", "setup_output_directory", 1872),
    ("src/main.py", "load_features_from_file", 1877),
    ("src/main.py", "drop_nan_rows", 1882),
    ("src/main.py", "convert_to_float32", 1887),
    ("src/main.py", "run_initial_backtest", 1892),
    ("src/main.py", "save_final_data", 1897),








    ("src/strategy.py", "run_backtest_simulation_v34", 1710),
    ("src/strategy.py", "initialize_time_series_split", 4045),
    ("src/strategy.py", "calculate_forced_entry_logic", 4048),
    ("src/strategy.py", "apply_kill_switch", 4051),
    ("src/strategy.py", "log_trade", 4054),
    ("src/strategy.py", "calculate_metrics", 2784),
    ("src/strategy.py", "aggregate_fold_results", 4057),


    ("ProjectP.py", "custom_helper_function", 20),
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
