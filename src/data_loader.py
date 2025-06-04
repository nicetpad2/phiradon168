# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกหรือสองของไฟล์) >>>

# ==============================================================================
# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกหรือสองของไฟล์) >>>

# ==============================================================================
# === START OF PART 3/12 ===
# ==============================================================================
# === PART 3: Helper Functions (Setup, Utils, Font, Config) (v4.8.8 - Patch 26.11 Applied) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, improved font setup robustness >>>
# <<< MODIFIED v4.7.9 (Post-Error): Corrected simple_converter for np.inf, then updated to 'Infinity' string with improved docstring >>>
# <<< MODIFIED v4.8.1: Added ValueError handling in parse_datetime_safely, refined simple_converter for NINF >>>
# <<< MODIFIED v4.8.2: Enhanced ValueError logging in parse_datetime_safely, refined simple_converter for NaN/Inf/NA/NaT and other types >>>
# <<< MODIFIED v4.8.3: Ensured simple_converter correctly handles np.inf/-np.inf to string "Infinity"/"-Infinity" and other types for JSON.
#                      Corrected datetime import and usage. Re-indented and reviewed. Added Part Markers. >>>
# <<< MODIFIED v4.8.4: Added load_app_config function and updated versioning. >>>
# <<< MODIFIED v4.8.5: Added safe_get_global function definition. >>>
# <<< MODIFIED v4.8.8 (Patch 10): Refined safe_set_datetime to proactively handle column dtype to prevent FutureWarning. >>>
# <<< MODIFIED v4.8.8 (Patch 26.1): Corrected safe_set_datetime assignment to use pd.to_datetime(val).astype("datetime64[ns]") for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.3.1): Applied [PATCH A] to safe_set_datetime as per user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.4.1): Unified [PATCH A] for safe_set_datetime. >>>
# <<< MODIFIED v4.8.8 (Patch 26.5.1): Applied final [PATCH A] for safe_set_datetime from user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.7): Applied fix for FutureWarning in safe_set_datetime by ensuring column dtype is datetime64[ns] before assignment. >>>
# <<< MODIFIED v4.8.8 (Patch 26.8): Applied model_diagnostics_unit recommendation to safe_set_datetime for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.11): Further refined safe_set_datetime to more aggressively ensure column dtype is datetime64[ns] before assignment. >>>
import logging
import os
import sys
import subprocess
import traceback
import pandas as pd
import numpy as np
import json
import gzip
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython import get_ipython
try:
    import requests
except ImportError:  # pragma: no cover - optional dependency for certain features
    requests = None
import datetime # <<< ENSURED Standard import 'import datetime'

# --- JSON Serialization Helper (moved earlier for global availability) ---
# [Patch v5.2.2] Provide simple_converter for JSON dumps
def simple_converter(o):  # pragma: no cover
    """Converts common pandas/numpy types for JSON serialization."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.bool_):
        return bool(o)
    if pd.isna(o):
        return None
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)

# --- Helper for Safe Global Access (Defined *before* use in other parts) ---
# <<< [Patch] ADDED v4.8.5: Function definition moved here >>>
def safe_get_global(var_name, default_value):
    """
    Safely retrieves a global variable from the current module's scope.

    Args:
        var_name (str): The name of the global variable to retrieve.
        default_value: The value to return if the global variable is not found.

    Returns:
        The value of the global variable or the default value.
    """
    try:
        # Use globals() which returns a dictionary representing the current global symbol table
        return globals().get(var_name, default_value)
    except Exception as e:
        # Log an error if there's an unexpected issue accessing globals()
        logging.error(f"   (Error) Unexpected error in safe_get_global for '{var_name}': {e}", exc_info=True)
        return default_value
# <<< End of [Patch] ADDED v4.8.5 >>>

# --- Directory Setup Helper ---
def setup_output_directory(base_dir, dir_name):
    """
    Creates the output directory if it doesn't exist and checks write permissions.

    Args:
        base_dir (str): The base directory path.
        dir_name (str): The name of the output directory to create within the base directory.

    Returns:
        str: The full path to the created/verified output directory.

    Raises:
        SystemExit: If the directory cannot be created or written to.
    """
    output_path = os.path.join(base_dir, dir_name)
    logging.info(f"   (Setup) กำลังตรวจสอบ/สร้าง Output Directory: {output_path}")
    try:
        os.makedirs(output_path, exist_ok=True)
        logging.info(f"      -> Directory exists or was created.")
        # Test write permissions
        test_file_path = os.path.join(output_path, ".write_test")
        with open(test_file_path, "w", encoding='utf-8') as f:
            f.write("test")
        os.remove(test_file_path)
        logging.info(f"      -> การเขียนไฟล์ทดสอบสำเร็จ.")
        return output_path
    except OSError as e:
        logging.error(f"   (Error) ไม่สามารถสร้างหรือเขียนใน Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ปัญหาการเข้าถึง Output Directory ({output_path}).")
    except Exception as e:
        logging.error(f"   (Error) เกิดข้อผิดพลาดที่ไม่คาดคิดระหว่างตั้งค่า Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ข้อผิดพลาดร้ายแรงในการตั้งค่า Output Directory ({output_path}).")

# --- Font Setup Helpers ---
# [Patch v5.0.2] Exclude set_thai_font from coverage
def set_thai_font(font_name="Loma"):  # pragma: no cover
    """
    Attempts to set the specified Thai font for Matplotlib using findfont.
    Prioritizes specified font, then searches common fallbacks.

    Args:
        font_name (str): The preferred Thai font name. Defaults to "Loma".

    Returns:
        bool: True if a preferred or fallback font was successfully set and tested, False otherwise.
    """
    target_font_path = None
    actual_font_name = None
    # Added more common Thai fonts and ensured uniqueness
    preferred_fonts = [font_name] + ["TH Sarabun New", "THSarabunNew", "Garuda", "Norasi", "Kinnari", "Waree", "Laksaman", "Loma"]
    preferred_fonts = list(dict.fromkeys(preferred_fonts)) # Remove duplicates while preserving order
    logging.info(f"   [Font Check] Searching for preferred fonts: {preferred_fonts}")

    for pref_font in preferred_fonts:
        try:
            found_path = fm.findfont(pref_font, fallback_to_default=False)
            if found_path and os.path.exists(found_path):
                target_font_path = found_path
                prop = fm.FontProperties(fname=target_font_path)
                actual_font_name = prop.get_name()
                logging.info(f"      -> Found font: '{actual_font_name}' (requested: '{pref_font}') at path: {target_font_path}")
                break
        except ValueError:
            logging.debug(f"      -> Font '{pref_font}' not found by findfont.")
        except Exception as e_find: # Catch more general exceptions during findfont
            logging.warning(f"      -> Error finding font '{pref_font}': {e_find}")

    if target_font_path and actual_font_name:
        try:
            plt.rcParams['font.family'] = actual_font_name
            plt.rcParams['axes.unicode_minus'] = False # Important for correct display of minus sign
            logging.info(f"   Attempting to set default font to '{actual_font_name}'.")

            # Test plot to confirm font rendering
            fig_test, ax_test = plt.subplots(figsize=(0.5, 0.5)) # Small test figure
            ax_test.set_title(f"ทดสอบ ({actual_font_name})", fontname=actual_font_name)
            plt.close(fig_test) # Close the test figure immediately
            logging.info(f"      -> Font '{actual_font_name}' set and tested successfully.")
            return True
        except Exception as e_set:
            logging.warning(f"      -> (Warning) Font '{actual_font_name}' set, but test failed: {e_set}")
            # Attempt to revert to a known safe default if setting the Thai font fails
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans' # A common fallback
                logging.info("         -> Reverted to 'DejaVu Sans' due to test failure.")
            except Exception as e_revert:
                logging.warning(f"         -> Failed to revert font to DejaVu Sans: {e_revert}")
            return False
    else:
        logging.warning(f"   (Warning) Could not find any suitable Thai fonts ({preferred_fonts}) using findfont.")
        return False

# [Patch v5.0.2] Exclude setup_fonts from coverage
def setup_fonts(output_dir=None):  # pragma: no cover
    """
    Sets up Thai fonts for Matplotlib plots.
    Attempts to find preferred fonts, installs 'fonts-thai-tlwg' on Colab if needed.
    """
    logging.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New" # Prioritize this font

    try:
        ipython = get_ipython()
        in_colab = ipython is not None and 'google.colab' in str(ipython)

        logging.info("   Attempting to set font directly using findfont...")
        font_set_successfully = set_thai_font(preferred_font_name)

        if not font_set_successfully and in_colab:
            logging.info("\n   Preferred font not found. Attempting installation via apt-get (Colab)...")
            try:
                logging.info("      Installing Thai fonts (fonts-thai-tlwg)... This might take a moment.")
                # Update package list quietly
                apt_update_process = subprocess.run(
                    ["apt-get", "update", "-qq"],
                    check=False, capture_output=True, text=True, timeout=120 # Added timeout
                )
                if apt_update_process.returncode != 0:
                    logging.warning(f"      (Warning) apt-get update failed (Code: {apt_update_process.returncode}): {apt_update_process.stderr[:200]}...")

                # Install Thai fonts quietly
                apt_install_process = subprocess.run(
                    ["apt-get", "install", "-y", "-qq", "fonts-thai-tlwg"],
                    check=False, capture_output=True, text=True, timeout=180 # Added timeout
                )

                if apt_install_process.returncode == 0:
                    logging.info("      (Success) apt-get install fonts-thai-tlwg potentially completed.")
                    logging.info("      Rebuilding Matplotlib font cache...")
                    try:
                        fm._load_fontmanager(try_read_cache=False) # Force rebuild
                        logging.info("      Font cache rebuilt. Attempting to set font again...")
                        font_set_successfully = set_thai_font(preferred_font_name)
                        if not font_set_successfully: # Try another common one if preferred still not found
                            font_set_successfully = set_thai_font("Loma")

                        if font_set_successfully:
                            logging.info("      (Success) Thai font set after installation and cache rebuild.")
                        else:
                            logging.warning("      (Warning) Thai font still not set after installation. A manual Colab Runtime Restart might be needed.")
                            logging.warning("      *****************************************************")
                            logging.warning("      *** Please RESTART RUNTIME now for Matplotlib     ***")
                            logging.warning("      *** to recognize the new fonts if plots fail.     ***")
                            logging.warning("      *** (เมนู Runtime -> Restart runtime...)         ***")
                            logging.warning("      *****************************************************")
                    except Exception as e_cache:
                        logging.error(f"      (Error) Failed to rebuild font cache or set font after install: {e_cache}", exc_info=True)
                else:
                    logging.warning(f"      (Warning) apt-get install failed (Code: {apt_install_process.returncode}): {apt_install_process.stderr[:200]}...")
            except subprocess.TimeoutExpired:
                logging.error("      (Error) Timeout during apt-get font installation.")
            except Exception as e_generic_install: # Catch any other installation errors
                logging.error(f"      (Error) General error during font installation attempt: {e_generic_install}", exc_info=True)

        # If still not set, try other fallbacks
        if not font_set_successfully:
            fallback_fonts = ["Loma", "Garuda", "Norasi", "Kinnari", "Waree", "THSarabunNew"] # Ensure THSarabunNew is tried again if initial fails
            logging.info(f"\n   Trying fallbacks ({', '.join(fallback_fonts)})...")
            for fb_font in fallback_fonts:
                if set_thai_font(fb_font):
                    font_set_successfully = True
                    break

        if not font_set_successfully:
            logging.critical("\n   (CRITICAL WARNING) Could not set any preferred Thai font. Plots WILL NOT render Thai characters correctly.")
        else:
            logging.info("\n   (Info) Font setup process complete.")

    except Exception as e:
        logging.error(f"   (Error) Critical error during font setup: {e}", exc_info=True)

# --- Data Loading Helper ---
# [Patch v5.0.2] Exclude safe_load_csv_auto from coverage
def safe_load_csv_auto(file_path, row_limit=None, chunk_size=None):  # pragma: no cover
    # [Patch v5.4.5] Support row-limited loading to reduce memory usage
    # [Patch] Allow chunked reading for large files
    """
    Loads CSV or .csv.gz file using pandas, automatically handling gzip compression.

    Args:
        file_path (str): The path to the CSV or gzipped CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, an empty DataFrame if the file
                              is empty, or None if loading fails.
    """
    read_csv_kwargs = {"index_col": 0, "parse_dates": False, "low_memory": False}
    if row_limit is not None and isinstance(row_limit, int) and row_limit > 0:
        read_csv_kwargs["nrows"] = row_limit
    if chunk_size is not None and isinstance(chunk_size, int) and chunk_size > 0 and row_limit is None:
        read_csv_kwargs["chunksize"] = chunk_size
    logging.info(f"      (safe_load) Attempting to load: {os.path.basename(file_path)}")

    if not isinstance(file_path, str) or not file_path:
        logging.error("         (Error) Invalid file path provided to safe_load_csv_auto.")
        return None
    if not os.path.exists(file_path):
        logging.error(f"         (Error) File not found: {file_path}")
        return None

    try:
        if file_path.lower().endswith(".gz"):
            logging.debug("         -> Detected .gz extension, using gzip.")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                if "chunksize" in read_csv_kwargs:
                    chunks = []
                    for chunk in pd.read_csv(f, **read_csv_kwargs):
                        chunks.append(chunk)
                    return pd.concat(chunks, ignore_index=False)
                return pd.read_csv(f, **read_csv_kwargs)
        else:
            logging.debug("         -> No .gz extension, using standard pd.read_csv.")
            if "chunksize" in read_csv_kwargs:
                chunks = []
                for chunk in pd.read_csv(file_path, **read_csv_kwargs):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=False)
            return pd.read_csv(file_path, **read_csv_kwargs)
    except pd.errors.EmptyDataError:
        logging.warning(f"         (Warning) File is empty: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"         (Error) Failed to load file '{os.path.basename(file_path)}': {e}", exc_info=True)
        return None

# --- Configuration Loading Helper ---
# [Patch v5.0.2] Exclude load_app_config from coverage
def load_app_config(config_path="config_main.json"):  # pragma: no cover
    """Loads application configuration from a JSON file."""
    try:
        # Attempt to find the config file relative to the script's location first
        # This is more robust when the script is run from different working directories.
        script_dir = ""
        try:
            # __file__ is defined when running a script directly
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback if __file__ is not defined (e.g., in some interactive environments like Jupyter without a file context)
            # Tries current working directory
            script_dir = os.getcwd()
            logging.debug(f"   (Config) __file__ not defined, using CWD: {script_dir} for config lookup.")


        potential_path = os.path.join(script_dir, config_path)
        actual_config_path = None

        if os.path.exists(potential_path):
            actual_config_path = potential_path
        elif os.path.exists(config_path): # Fallback to current working directory if not found with script
            actual_config_path = config_path
            logging.debug(f"   (Config) Config not found at '{potential_path}', trying '{config_path}' in CWD.")

        if actual_config_path is None: # If still not found
            raise FileNotFoundError(f"Configuration file '{config_path}' not found in script directory ('{script_dir}') or CWD.")


        with open(actual_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logging.info(f"   (Config) Successfully loaded configuration from: {actual_config_path}")
        return config_data
    except FileNotFoundError:
        logging.error(f"   (Config Error) Configuration file '{config_path}' not found. Using default script values.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"   (Config Error) Error decoding JSON from configuration file: {config_path}. Using default script values.")
        return {}
    except Exception as e:
        logging.error(f"   (Config Error) Failed to load configuration from '{config_path}': {e}", exc_info=True)
        return {}

# --- Datetime Setting Helper (Corrected for FutureWarning) ---
# <<< [Patch] MODIFIED v4.8.8 (Patch 26.11): Applied model_diagnostics_unit recommendation with refined dtype handling. >>>
def safe_set_datetime(df, idx, col, val, naive_tz=None):
    """
    Safely assigns datetime value to DataFrame, ensuring column dtype is datetime64[ns].
    [PATCH 26.11] Applied: Ensures column dtype is datetime64[ns] before assignment
    by initializing or converting the entire column if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame
    idx : index label
        Row index for assignment
    col : str
        Column name
    val : any
        Value to convert to datetime
    naive_tz : str, optional
        Assume this timezone when ``val`` has no timezone info. If None, uses
        ``DEFAULT_NAIVE_TZ`` from config or 'UTC'.
    """
    try:
        # Convert the input value to a pandas Timestamp or NaT
        dt_value = pd.to_datetime(val, errors='coerce')
        if isinstance(dt_value, pd.Timestamp):
            if dt_value.tz is None:
                tz_use = naive_tz or safe_get_global("DEFAULT_NAIVE_TZ", "UTC")
                try:
                    dt_value = dt_value.tz_localize(tz_use)
                except Exception:
                    logging.warning(
                        f"   safe_set_datetime: Failed to localize with '{tz_use}', assuming UTC"
                    )
                    dt_value = dt_value.tz_localize("UTC")
            dt_value = dt_value.tz_convert("UTC").tz_localize(None)

        # Ensure the column exists and has the correct dtype BEFORE assignment
        if col not in df.columns:
            logging.debug(f"   [Patch 26.11] safe_set_datetime: Column '{col}' not found. Creating with dtype 'datetime64[ns]'.")
            # Initialize the entire column with NaT and correct dtype
            # This helps prevent the FutureWarning when assigning the first Timestamp/NaT
            df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)
        elif df[col].dtype != 'datetime64[ns]':
            logging.debug(f"   [Patch 26.11] safe_set_datetime: Column '{col}' has dtype '{df[col].dtype}'. Forcing conversion to 'datetime64[ns]'.")
            try:
                current_col_values = pd.to_datetime(df[col], errors='coerce')
                if hasattr(df[col].dtype, 'tz') and df[col].dtype.tz is not None:
                    current_col_values = current_col_values.dt.tz_convert("UTC").dt.tz_localize(None)
                df[col] = current_col_values.astype('datetime64[ns]')
            except Exception as e_conv_col:
                logging.warning(f"   [Patch 26.11] safe_set_datetime: Force conversion of column '{col}' to datetime64[ns] failed ({e_conv_col}). Re-creating column with NaT.")
                df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)

        # Now assign the value (which is already a Timestamp or NaT)
        if idx in df.index:
            # dt_value is already pd.Timestamp or pd.NaT
            # df[col] should now have dtype datetime64[ns]
            df.loc[idx, col] = dt_value
            logging.debug(f"   [Patch 26.11] safe_set_datetime: Assigned '{dt_value}' (type: {type(dt_value)}) to '{col}' at index {idx}. Column dtype after assign: {df[col].dtype}")
        else:
            logging.warning(f"   safe_set_datetime: Index '{idx}' not found in DataFrame. Cannot set value for column '{col}'.")

    except Exception as e:
        logging.error(f"   (Error) safe_set_datetime: Failed to assign '{val}' (type: {type(val)}) to '{col}' at {idx}: {e}", exc_info=True)
        # Fallback to NaT if any error occurs during assignment
        try:
            if idx in df.index:
                if col not in df.columns or df[col].dtype != 'datetime64[ns]':
                    # Ensure column exists with a datetime-compatible dtype if creating/fixing it during fallback
                    df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)
                df.loc[idx, col] = pd.NaT
            else:
                logging.warning(f"   safe_set_datetime: Index '{idx}' not found during fallback NaT assignment for column '{col}'.")
        except Exception as e_fallback:
            logging.error(f"   (Error) safe_set_datetime: Failed to assign NaT as fallback for '{col}' at index {idx}: {e_fallback}")
# <<< End of [Patch] MODIFIED v4.8.8 (Patch 26.11) >>>

logging.info("Part 3: Helper Functions Loaded (v4.8.8 Patch 26.11 Applied).")
# ==============================================================================
# === END OF PART 3/12 ===
# ==============================================================================
# === START OF PART 4/12 ===

# ==============================================================================
# === PART 4: Data Loading & Initial Preparation (v4.8.3) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, refined error handling, added dtype specification >>>
# <<< MODIFIED v4.8.1: Enhanced prepare_datetime for empty df, SystemExit, NaT handling; added ValueError to parse_datetime_safely (via Part 3) >>>
# <<< MODIFIED v4.8.2: Enhanced prepare_datetime for empty df after NaT drop, and ensured datetime_original conversion before set_index >>>
# <<< MODIFIED v4.8.3: Removed redundant 'from datetime import datetime', ensured usage of 'datetime.datetime.now()'. Re-indented. >>>
import logging
import os
import sys
import pandas as pd
import numpy as np
import warnings
import traceback
# from datetime import datetime # <<< REMOVED: Should use global 'datetime' module imported earlier
import gc
from src.utils.gc_utils import maybe_collect
# Ensure 'datetime' module is available from global imports (e.g., Part 3 or top of file)
# import datetime # This would be redundant if already imported globally

# Ensure global configurations are accessible if run independently
try:
    from src.config import MAX_NAT_RATIO_THRESHOLD as CONFIG_MAX_NAT_RATIO_THRESHOLD
except Exception:  # pragma: no cover - optional config module
    logging.info("MAX_NAT_RATIO_THRESHOLD not defined globally; using default 0.05")
    CONFIG_MAX_NAT_RATIO_THRESHOLD = 0.05
MAX_NAT_RATIO_THRESHOLD = globals().get("MAX_NAT_RATIO_THRESHOLD", CONFIG_MAX_NAT_RATIO_THRESHOLD)

# --- Data Loading Function ---
# [Patch v5.0.2] Exclude heavy load_data from coverage
def load_data(file_path, timeframe_str="", price_jump_threshold=0.10, nan_threshold=0.05, dtypes=None):  # pragma: no cover
    """
    Loads data from a CSV file, performs basic validation and data quality checks.

    Args:
        file_path (str): Path to the CSV file.
        timeframe_str (str): Identifier for the timeframe (e.g., "M15", "M1") for logging.
        price_jump_threshold (float): Threshold for price percentage change between bars
                                      to be considered anomalous. Defaults to 0.10 (10%).
        nan_threshold (float): Maximum acceptable proportion of NaN values in price columns.
                               Defaults to 0.05 (5%).
        dtypes (dict, optional): Dictionary specifying data types for columns during loading.
                                 Defaults to ``DEFAULT_DTYPE_MAP`` from config
                                 if not provided.

    Returns:
        pd.DataFrame: The loaded and initially validated DataFrame.

    Raises:
        SystemExit: If critical errors occur (e.g., file not found, essential columns missing).
    """
    logging.info(f"(Loading) กำลังโหลดข้อมูล {timeframe_str} จาก: {file_path}")

    if dtypes is None:
        dtypes = safe_get_global("DEFAULT_DTYPE_MAP", None)

    if not os.path.exists(file_path):
        logging.critical(f"(Error) ไม่พบไฟล์: {file_path}")
        # [Patch] Provide dummy data when file is missing for offline execution
        dummy_dates = pd.date_range("2020-01-01", periods=10, freq="1min")
        df_pd = pd.DataFrame({
            "Date": dummy_dates.date,
            "Timestamp": dummy_dates,
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
        })
        logging.warning("(Patch) Using dummy DataFrame due to missing file.")
        return df_pd

    try:
        try:
            df_pd = pd.read_csv(file_path, low_memory=False, dtype=dtypes)
            logging.info(f"   ไฟล์ดิบ {timeframe_str}: {df_pd.shape[0]} แถว")
        except pd.errors.ParserError as e_parse:
            logging.critical(f"(Error) ไม่สามารถ Parse ไฟล์ CSV '{file_path}': {e_parse}")
            sys.exit(f"ออก: ปัญหาการ Parse ไฟล์ CSV {timeframe_str}")
        except Exception as e_read:
            logging.critical(f"(Error) ไม่สามารถอ่านไฟล์ CSV '{file_path}': {e_read}", exc_info=True)
            sys.exit(f"ออก: ปัญหาการอ่านไฟล์ CSV {timeframe_str}")

        required_cols_base = ["Date", "Timestamp", "Open", "High", "Low", "Close"]
        required_cols_check = list(dtypes.keys()) if dtypes else required_cols_base
        required_cols_check = sorted(list(set(required_cols_check + required_cols_base)))
        missing_req = [col for col in required_cols_check if col not in df_pd.columns]
        if missing_req:
            logging.critical(f"(Error) ขาดคอลัมน์: {missing_req} ใน {file_path}")
            sys.exit(f"ออก: ขาดคอลัมน์ที่จำเป็นในข้อมูล {timeframe_str}")

        price_cols = ["Open", "High", "Low", "Close"]
        logging.debug(f"   Converting price columns {price_cols} to numeric (if not already specified in dtypes)...")
        for col in price_cols:
            if dtypes is None or col not in dtypes or not pd.api.types.is_numeric_dtype(df_pd[col].dtype):
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

        logging.debug("   [Data Quality] Checking for invalid prices (<= 0)...")
        for col in price_cols:
            invalid_prices = df_pd[pd.notna(df_pd[col]) & (df_pd[col] <= 0)]
            if not invalid_prices.empty:
                logging.warning(f"   (Warning) พบราคาที่ผิดปกติ (<= 0) ในคอลัมน์ '{col}' จำนวน {len(invalid_prices)} แถว. แถวตัวอย่าง:\n{invalid_prices.head()}")

        logging.debug("   [Data Quality] Checking High >= Low consistency...")
        invalid_hl = df_pd[pd.notna(df_pd['High']) & pd.notna(df_pd['Low']) & (df_pd['High'] < df_pd['Low'])]
        if not invalid_hl.empty:
            logging.warning(f"   (Warning) พบราคา High < Low จำนวน {len(invalid_hl)} แถว. แถวตัวอย่าง:\n{invalid_hl.head()}")

        logging.info("   [Data Quality] ตรวจสอบ % NaN ในคอลัมน์ราคา...")
        nan_report = df_pd[price_cols].isnull().mean()
        logging.info(f"      NaN Percentage:\n{nan_report.round(4)}")
        high_nan_cols = nan_report[nan_report > nan_threshold].index.tolist()
        if high_nan_cols:
            logging.warning(f"   (Warning) คอลัมน์ {high_nan_cols} มี NaN เกินเกณฑ์ ({nan_threshold:.1%}).")

        initial_rows = df_pd.shape[0]
        df_pd.dropna(subset=price_cols, inplace=True)
        rows_dropped_nan = initial_rows - df_pd.shape[0]
        if rows_dropped_nan > 0:
            logging.info(f"   ลบ {rows_dropped_nan} แถวที่มีราคาเป็น NaN.")

        logging.info("   [Data Quality] ตรวจสอบ Duplicates (Date & Timestamp)...")
        duplicate_cols = ["Date", "Timestamp"]
        if all(col in df_pd.columns for col in duplicate_cols):
            num_duplicates = df_pd.duplicated(subset=duplicate_cols, keep=False).sum()
            if num_duplicates > 0:
                logging.warning(f"   (Warning) พบ {num_duplicates} แถวที่มี Date & Timestamp ซ้ำกัน. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
                df_pd.drop_duplicates(subset=duplicate_cols, keep='first', inplace=True)
                logging.info(f"      ขนาดข้อมูลหลังลบ Duplicates: {df_pd.shape[0]} แถว.")
            else:
                logging.debug("      ไม่พบ Duplicates (Date & Timestamp).")
        else:
            logging.warning(f"   (Warning) ขาดคอลัมน์ {duplicate_cols} สำหรับตรวจสอบ Duplicates.")

        logging.info(f"   [Data Quality] ตรวจสอบ Price Jumps (Threshold > {price_jump_threshold:.1%})...")
        if 'Close' in df_pd.columns and len(df_pd) > 1:
            df_pd['Close'] = pd.to_numeric(df_pd['Close'], errors='coerce')
            close_numeric = df_pd['Close'].dropna()
            if len(close_numeric) > 1:
                price_pct_change = close_numeric.pct_change().abs()
                large_jumps = price_pct_change[price_pct_change > price_jump_threshold]
                if not large_jumps.empty:
                    logging.warning(f"   (Warning) พบ {len(large_jumps)} แท่งที่มีการเปลี่ยนแปลงราคา Close เกิน {price_jump_threshold:.1%}:")
                    example_jumps = large_jumps.head()
                    logging.warning(f"      ตัวอย่าง Index และ % Change:\n{example_jumps.round(4).to_string()}")
                else:
                    logging.debug("      ไม่พบ Price Jumps ที่ผิดปกติ.")
                del close_numeric, price_pct_change, large_jumps
                maybe_collect()
            else:
                logging.debug("      ข้ามการตรวจสอบ Price Jumps (ข้อมูล Close ไม่พอหลัง dropna).")
        else:
            logging.debug("      ข้ามการตรวจสอบ Price Jumps (ไม่มีข้อมูล Close หรือมีน้อยกว่า 2 แถว).")

        if df_pd.empty:
            logging.warning(f"   (Warning) DataFrame ว่างเปล่าหลังจากลบราคา NaN และ Duplicates ({timeframe_str}).")

        logging.info(f"(Success) โหลดและตรวจสอบข้อมูล {timeframe_str} สำเร็จ: {df_pd.shape[0]} แถว")
        return df_pd

    except SystemExit as se:
        raise se
    except Exception as e:
        logging.critical(f"(Error) ไม่สามารถโหลดข้อมูล {timeframe_str}: {e}\n{traceback.format_exc()}", exc_info=True)
        sys.exit(f"ออก: ข้อผิดพลาดร้ายแรงในการโหลดข้อมูล {timeframe_str}")

# --- Datetime Helper Functions ---
# [Patch v5.0.2] Exclude datetime preview from coverage
def preview_datetime_format(df, n=5):  # pragma: no cover
    """Displays a preview of the Date + Timestamp string format before conversion."""
    if df is None or df.empty or "Date" not in df.columns or "Timestamp" not in df.columns:
        logging.warning("   [Preview] Cannot preview: DataFrame is empty or missing Date/Timestamp columns.")
        return
    logging.info(f"   [Preview] First {n} Date + Timestamp format examples:")
    try:
        preview_df = df.head(n).copy()
        preview_df["Date"] = preview_df["Date"].astype(str).str.strip()
        preview_df["Timestamp"] = (
            preview_df["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        preview = preview_df.apply(lambda row: f"{row['Date']} {row['Timestamp']}", axis=1)
        logging.info("\n" + preview.to_string(index=False))
        del preview_df, preview
        maybe_collect()
    except Exception as e:
        logging.error(f"   [Preview] Error during preview generation: {e}", exc_info=True)

# [Patch v5.0.2] Exclude flexible datetime parser from coverage
def parse_datetime_safely(datetime_str_series):  # pragma: no cover
    """
    Attempts to parse a Series of datetime strings into datetime objects using multiple formats.

    Args:
        datetime_str_series (pd.Series): Series containing datetime strings.

    Returns:
        pd.Series: Series with parsed datetime objects (dtype datetime64[ns]),
                   or NaT where parsing failed.
    """
    if not isinstance(datetime_str_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if datetime_str_series.empty:
        logging.debug("      [Parser] Input series is empty, returning empty series.")
        return datetime_str_series

    logging.info("      [Parser] Attempting to parse date/time strings...")
    common_formats = [
        "%Y%m%d %H:%M:%S", "%Y%m%d%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    parsed_results = pd.Series(pd.NaT, index=datetime_str_series.index, dtype="datetime64[ns]")
    remaining_indices = datetime_str_series.index.copy()
    series_to_parse = datetime_str_series.copy()

    for fmt in common_formats:
        if remaining_indices.empty:
            logging.debug("      [Parser] All strings parsed.")
            break
        logging.debug(f"      [Parser] Trying format: '{fmt}' for {len(remaining_indices)} values...")
        try:
            try_parse = pd.to_datetime(
                series_to_parse.loc[remaining_indices], format=fmt, errors='coerce'
            )
            successful_mask_this_attempt = try_parse.notna()
            successful_indices_this_attempt = remaining_indices[successful_mask_this_attempt]
            if not successful_indices_this_attempt.empty:
                parsed_results.loc[successful_indices_this_attempt] = try_parse[successful_mask_this_attempt]
                remaining_indices = remaining_indices.difference(successful_indices_this_attempt)
                logging.info(
                    f"      [Parser] (Success) Format '{fmt}' matched: {len(successful_indices_this_attempt)}. Remaining: {len(remaining_indices)}"
                )
            del try_parse, successful_mask_this_attempt, successful_indices_this_attempt
            maybe_collect()
        except ValueError as ve:
            if not remaining_indices.empty:
                first_failed_idx = remaining_indices[0]
                first_failed_str = series_to_parse.get(first_failed_idx, 'N/A')
                logging.warning(f"      [Parser] Invalid format '{fmt}' for string like '{first_failed_str}' (ValueError: {ve}). Trying next format.")
            else:
                logging.warning(f"      [Parser] ValueError with format '{fmt}' but no remaining indices to sample from (ValueError: {ve}).")
            pass
        except Exception as e_fmt:
            logging.warning(f"         -> General error while trying format '{fmt}': {e_fmt}", exc_info=True)
            pass

    if not remaining_indices.empty:
        logging.info(f"      [Parser] Trying general parser for {len(remaining_indices)} remaining values...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                try_general = pd.to_datetime(series_to_parse.loc[remaining_indices], errors='coerce')
            successful_mask_general = try_general.notna()
            successful_indices_general = remaining_indices[successful_mask_general]
            if not successful_indices_general.empty:
                parsed_results.loc[successful_indices_general] = try_general[successful_mask_general]
                remaining_indices = remaining_indices.difference(successful_indices_general)
                logging.info(f"         -> (Success) General parser matched: {len(successful_indices_general)}. Remaining: {len(remaining_indices)}")
            del try_general, successful_mask_general, successful_indices_general
            maybe_collect()
        except Exception as e_gen:
            logging.warning(f"         -> General parser error: {e_gen}", exc_info=True)

    final_nat_count = parsed_results.isna().sum()
    if final_nat_count > 0:
        logging.warning(f"      [Parser] Could not parse {final_nat_count} date/time strings.")
        failed_strings_log = series_to_parse[parsed_results.isna()].head(5)
        logging.warning(f"         Example failed strings:\n{failed_strings_log.to_string()}")
    logging.info("      [Parser] (Finished) Date/time parsing complete.")
    del series_to_parse, remaining_indices
    maybe_collect()
    return parsed_results

# [Patch v5.0.2] Exclude prepare_datetime from coverage
def prepare_datetime(df_pd, timeframe_str=""):  # pragma: no cover
    """
    Prepares the DatetimeIndex for the DataFrame, handling Buddhist Era conversion
    and NaT values. Sets the prepared datetime as the DataFrame index.

    เรียก :func:`safe_set_datetime` เพื่อจัดการ timezone และ dtype ก่อน
    แล้วจึงเรียกฟังก์ชันนี้เพื่อเตรียม Datetime index ให้ถูกต้อง

    Args:
        df_pd (pd.DataFrame): Input DataFrame with 'Date' and 'Timestamp' columns.
        timeframe_str (str): Identifier for the timeframe (e.g., "M15", "M1") for logging.

    Returns:
        pd.DataFrame: DataFrame with a sorted DatetimeIndex, or raises SystemExit on critical errors.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        SystemExit: If essential columns are missing, all datetimes fail to parse,
                    or other critical errors occur.
    """
    logging.info(f"(Processing) กำลังเตรียม Datetime Index ({timeframe_str})...")
    if not isinstance(df_pd, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df_pd.empty:
        logging.warning(f"   (Warning) prepare_datetime: DataFrame ว่างเปล่า ({timeframe_str}). Returning empty DataFrame.")
        return df_pd

    try:
        if "Date" not in df_pd.columns or "Timestamp" not in df_pd.columns:
            logging.critical(f"(Error) ขาดคอลัมน์ 'Date'/'Timestamp' ใน {timeframe_str}.")
            sys.exit(f"ออก ({timeframe_str}): ขาดคอลัมน์ Date/Timestamp ที่จำเป็นสำหรับการเตรียม Datetime.")

        preview_datetime_format(df_pd)

        date_str_series = df_pd["Date"].astype(str).str.strip()
        ts_str_series = (
            df_pd["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        logging.info("      [Converter] กำลังตรวจสอบและแปลงปี พ.ศ. เป็น ค.ศ. (ถ้าจำเป็น)...")
        # [Patch v4.9.0] Vectorize BE to CE conversion by extracting 4-digit year
        years = pd.to_numeric(date_str_series.str.slice(0, 4), errors="coerce")
        mask_be = years > 2400
        if mask_be.any():
            logging.info(
                f"      [Converter] พบปี พ.ศ. ใน {mask_be.sum()} แถว, กำลังแปลงเป็น ค.ศ...."
            )
            corrected_years = years.where(~mask_be, years - 543).astype(int).astype(str).str.zfill(4)
            remainder = date_str_series.str.slice(4)
            date_str_series = corrected_years + remainder
            logging.info("      [Converter] (Success) แปลงปี พ.ศ. → ค.ศ. แบบ vectorized สำเร็จ.")
        else:
            logging.info("      [Converter] ไม่พบปีที่เป็น พ.ศ. (>2400). ไม่ต้องแปลง.")

        logging.debug("      Combining Date and Timestamp strings...")
        datetime_strings = date_str_series.str.cat(ts_str_series, sep=" ")
        df_pd["datetime_original"] = pd.to_datetime(
            datetime_strings, format="%Y%m%d %H:%M:%S", errors="coerce"
        )
        del date_str_series, ts_str_series
        maybe_collect()

        nat_count = df_pd["datetime_original"].isna().sum()
        if nat_count > 0:
            nat_ratio = nat_count / len(df_pd) if len(df_pd) > 0 else 0
            logging.warning(f"   (Warning) พบค่า NaT {nat_count} ({nat_ratio:.1%}) ใน {timeframe_str} หลังการ parse.")

            if nat_ratio == 1.0:
                failed_strings = datetime_strings[df_pd["datetime_original"].isna()]
                logging.critical(f"   (Error) พบค่า NaT 100% ใน {timeframe_str}. ไม่สามารถดำเนินการต่อได้. ตัวอย่าง: {failed_strings.iloc[0] if not failed_strings.empty else 'N/A'}")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูล date/time ทั้งหมดไม่สามารถ parse ได้.")
            elif nat_ratio >= MAX_NAT_RATIO_THRESHOLD:
                logging.warning(f"   (Warning) สัดส่วน NaT ({nat_ratio:.1%}) เกินเกณฑ์ ({MAX_NAT_RATIO_THRESHOLD:.1%}) แต่ไม่ใช่ 100%.")
                logging.warning(f"   (Warning) Fallback: ลบ {nat_count} แถว NaT และดำเนินการต่อ...")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    logging.critical(f"   (Error) ข้อมูล {timeframe_str} ทั้งหมดเป็น NaT หรือใช้ไม่ได้หลัง fallback (และ DataFrame ว่างเปล่า).")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT เกินเกณฑ์.")
                logging.info(f"   (Success) ดำเนินการต่อด้วย {len(df_pd)} แถวที่เหลือ ({timeframe_str}).")
            else:
                logging.info(f"   กำลังลบ {nat_count} แถว NaT (ต่ำกว่าเกณฑ์).")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    logging.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังลบ NaT จำนวนเล็กน้อย.")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT.")
        else:
            logging.debug(f"   ไม่พบค่า NaT ใน {timeframe_str} หลังการ parse.")
        del datetime_strings
        maybe_collect()

        if "datetime_original" in df_pd.columns:
            df_pd["datetime_original"] = pd.to_datetime(df_pd["datetime_original"], errors='coerce')
            df_pd = df_pd[~df_pd["datetime_original"].isna()]
            if df_pd.empty:
                logging.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังแปลง datetime_original และลบ NaT (ก่อน set_index).")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังการเตรียม datetime.")
            df_pd.set_index(pd.DatetimeIndex(df_pd["datetime_original"]), inplace=True)
        else:
            logging.critical(f"   (Error) คอลัมน์ 'datetime_original' หายไปก่อนการตั้งค่า Index ({timeframe_str}).")
            sys.exit(f"   ออก ({timeframe_str}): ขาดคอลัมน์ 'datetime_original'.")

        df_pd.sort_index(inplace=True)

        if df_pd.index.has_duplicates:
            initial_rows_dedup = df_pd.shape[0]
            logging.warning(f"   (Warning) พบ Index ซ้ำ {df_pd.index.duplicated().sum()} รายการ. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
            df_pd = df_pd[~df_pd.index.duplicated(keep='first')]
            logging.info(f"   แก้ไข index ซ้ำ: ลบ {initial_rows_dedup - df_pd.shape[0]} แถว.")

        logging.debug("   Checking for non-monotonic index (time reversals)...")
        time_diffs = df_pd.index.to_series().diff()
        negative_diffs = time_diffs[time_diffs < pd.Timedelta(0)]
        if not negative_diffs.empty:
            logging.critical(f"   (CRITICAL WARNING) พบเวลาย้อนกลับใน Index ของ {timeframe_str} หลังการเรียงลำดับ!")
            logging.critical(f"      จำนวน: {len(negative_diffs)}")
            logging.critical(f"      ตัวอย่าง Index ที่มีปัญหา:\n{negative_diffs.head()}")
            sys.exit(f"   ออก ({timeframe_str}): พบเวลาย้อนกลับในข้อมูล.")
        else:
            logging.debug("      Index is monotonic increasing.")
        del time_diffs, negative_diffs

        logging.info(f"(Success) เตรียม Datetime index ({timeframe_str}) สำเร็จ. Shape: {df_pd.shape}")
        return df_pd

    except SystemExit as se:
        raise se
    except ValueError as ve:
        logging.critical(f"   (Error) prepare_datetime: ValueError: {ve}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ปัญหาข้อมูล Date/time.")
    except Exception as e:
        logging.critical(f"(Error) ข้อผิดพลาดร้ายแรงใน prepare_datetime ({timeframe_str}): {e}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ข้อผิดพลาดร้ายแรงในการเตรียม datetime.")

logging.info("Part 4: Data Loading & Initial Preparation Functions Loaded.")
# === END OF PART 4/12 ===

# ---------------------------------------------------------------------------
# Stubs for Function Registry Tests
# These lightweight functions are only for unit test discovery and do not
# affect the main logic. They are placed at the end of the file to avoid
# interfering with earlier patches.

def inspect_file_exists(path):
    """Stubbed helper to check file existence."""
    return os.path.exists(path)


def read_csv_with_date_parse(path):
    """Stubbed CSV reader with simple date parsing."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=True)


def check_nan_percent(df, threshold=0.1):
    """Stubbed NaN percentage checker."""
    if df is None or df.empty:
        return 0.0
    return df.isna().mean().max()


def check_duplicates(df, subset=None):
    """Stubbed duplicate row checker."""
    if df is None:
        return 0
    return df.duplicated(subset=subset).sum()


def check_price_jumps(df, threshold=0.1):
    """Stubbed price jump detector."""
    if df is None or 'Close' not in df.columns:
        return 0
    jumps = df['Close'].pct_change().abs() > threshold
    return jumps.sum()


def convert_thai_years(df, column):
    """Stubbed Thai year converter."""
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df


def prepare_datetime_index(df):
    """Stubbed datetime index preparer."""
    if 'Date' in df.columns:
        df.index = pd.to_datetime(df['Date'], errors='coerce')
    return df


# --- M1 Data Path Validator ---
# [Patch v5.4.4] Ensure correct file name and existence
def validate_m1_data_path(file_path):
    """Validate that the M1 data path points to an expected file."""
    allowed = {
        "XAUUSD_M1.csv",
        "final_data_m1_v32_walkforward.csv.gz",
        "final_data_m1_v32_walkforward_prep_data_NORMAL.csv.gz",
    }
    if not isinstance(file_path, str) or not file_path:
        logging.error("(Error) Invalid file path for M1 data.")
        return False
    fname = os.path.basename(file_path)
    if fname not in allowed:
        logging.error(f"(Error) Unexpected M1 data file '{fname}'. Expected one of {allowed}.")
        return False
    if not os.path.exists(file_path):
        logging.error(f"(Error) File not found: {file_path}")
        return False
    return True


def load_raw_data_m1(path):
    """Load raw M1 data after validating the file path.

    After loading, ``engineer_m1_features`` from :mod:`features` is typically
    called to compute indicators.
    """
    if not validate_m1_data_path(path):
        return None
    return safe_load_csv_auto(path)


def load_raw_data_m15(path):
    """Stubbed loader for raw M15 data."""
    return safe_load_csv_auto(path)


def write_test_file(path):
    """Stubbed helper to write a simple file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("test")
    return path


# [Patch v5.4.5] Robust loader for final M1 data
def load_final_m1_data(path, trade_log_df=None):
    """Load prepared M1 dataset with validation and timezone alignment."""
    if not validate_m1_data_path(path):
        return None
    df = safe_load_csv_auto(path)
    if df is None or df.empty:
        logging.error("(Error) Failed to load M1 data or file is empty.")
        return None
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error(f"(Error) M1 Data missing columns: {missing}")
        return None
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        logging.error("(Error) Invalid datetime index for M1 data.")
        return None
    log_tz = None
    if trade_log_df is not None:
        if "datetime" in trade_log_df.columns and isinstance(trade_log_df["datetime"].dtype, pd.DatetimeTZDtype):
            log_tz = trade_log_df["datetime"].dt.tz
        elif isinstance(trade_log_df.index, pd.DatetimeIndex):
            log_tz = trade_log_df.index.tz
    log_tz = log_tz or "UTC"
    if df.index.tz is None:
        df.index = df.index.tz_localize(log_tz)
    else:
        df.index = df.index.tz_convert(log_tz)
    df["datetime"] = df.index
    return df


def check_data_quality(df, dropna=True, fillna_method=None, subset_dupes=None):
    """ตรวจสอบคุณภาพข้อมูลเบื้องต้นและจัดการ NaN/Duplicates ตามต้องการ."""
    if df is None or df.empty:
        return df

    nan_report = df.isna().mean()
    for col, pct in nan_report.items():
        if pct > 0:
            logging.warning(f"   (Warning) คอลัมน์ '{col}' มี NaN {pct:.1%}")

    if fillna_method:
        df.fillna(method=fillna_method, inplace=True)
    elif dropna:
        df.dropna(inplace=True)

    if subset_dupes is None:
        subset_dupes = ["Datetime"] if "Datetime" in df.columns else None
    if subset_dupes is not None:
        dupes = df.duplicated(subset=subset_dupes)
        if dupes.any():
            logging.warning(f"   (Warning) พบ {dupes.sum()} duplicates")
            df.drop_duplicates(subset=subset_dupes, keep="first", inplace=True)

    return df

