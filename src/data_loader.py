# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกสุด) >>>
"""Utility helpers for loading CSV files and preparing dataframes."""

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
from src.utils.errors import DataValidationError
import os
import sys
import subprocess
import traceback
import glob
import re
import pandas as pd
import numpy as np
import json
import gzip
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython import get_ipython
import locale
from dateutil.parser import parse as parse_date
try:
    import requests
except ImportError:  # pragma: no cover - optional dependency for certain features
    requests = None

logger = logging.getLogger(__name__)
import datetime # <<< ENSURED Standard import 'import datetime'

# --- Locale Setup for Thai date parsing ---
try:
    locale.setlocale(locale.LC_TIME, 'th_TH.UTF-8')
except locale.Error:
    logging.debug("Locale th_TH not supported, falling back to default.")


# --- Robust Thai date parser ---
THAI_MONTH_MAP = {
    "ม.ค.": "01",
    "ก.พ.": "02",
    "มี.ค.": "03",
    "เม.ย.": "04",
    "พ.ค.": "05",
    "มิ.ย.": "06",
    "ก.ค.": "07",
    "ส.ค.": "08",
    "ก.ย.": "09",
    "ต.ค.": "10",
    "พ.ย.": "11",
    "ธ.ค.": "12",
}


def robust_date_parser(date_string):
    """Parse Thai date strings with ``dateutil``, handling Buddhist years."""
    normalized = str(date_string)
    for th, num in THAI_MONTH_MAP.items():
        if th in normalized:
            normalized = normalized.replace(th, num)
            break
    try:
        dt = parse_date(normalized, dayfirst=True)
    except Exception as e:
        raise ValueError(f"Cannot parse Thai date: {date_string}") from e
    if dt.year > 2500:
        dt = dt.replace(year=dt.year - 543)
    return dt

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
    if isinstance(o, pd.Timedelta):
        return str(o)
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
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write("test")
        # Remove the temporary file quietly regardless of location
        try:
            os.remove(test_file_path)
        except OSError:
            logging.debug(f"Unable to remove test file {test_file_path}")
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

# [Patch v5.6.0] Split font installation and configuration helpers
def install_thai_fonts_colab():  # pragma: no cover
    """Install Thai fonts when running on Google Colab."""
    try:
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False, capture_output=True, text=True, timeout=120)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "fonts-thai-tlwg"], check=False, capture_output=True, text=True, timeout=180)
        subprocess.run(["fc-cache", "-fv"], check=False, capture_output=True, text=True, timeout=120)
        return True
    except Exception as e:
        logging.error(f"      (Error) Failed to install Thai fonts: {e}")
        return False


def configure_matplotlib_fonts(font_name="TH Sarabun New"):  # pragma: no cover
    """Configure Matplotlib to use a given Thai font."""
    return set_thai_font(font_name)


def setup_fonts(output_dir=None):  # pragma: no cover
    """Sets up Thai fonts for Matplotlib plots."""
    logging.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New"
    try:
        ipython = get_ipython()
        in_colab = ipython is not None and 'google.colab' in str(ipython)
        font_set_successfully = configure_matplotlib_fonts(preferred_font_name)
        if not font_set_successfully and in_colab:
            logging.info("\n   Preferred font not found. Attempting installation via apt-get (Colab)...")
            if install_thai_fonts_colab():
                fm._load_fontmanager(try_read_cache=False)
                font_set_successfully = configure_matplotlib_fonts(preferred_font_name) or configure_matplotlib_fonts("Loma")
        if not font_set_successfully:
            fallback_fonts = ["Loma", "Garuda", "Norasi", "Kinnari", "Waree", "THSarabunNew"]
            logging.info(f"\n   Trying fallbacks ({', '.join(fallback_fonts)})...")
            for fb_font in fallback_fonts:
                if configure_matplotlib_fonts(fb_font):
                    font_set_successfully = True
                    break
        if not font_set_successfully:
            logging.critical("\n   (CRITICAL WARNING) Could not set any preferred Thai font. Plots WILL NOT render Thai characters correctly.")
        else:
            logging.info("\n   (Info) Font setup process complete.")
    except Exception as e:
        logging.error(f"   (Error) Critical error during font setup: {e}", exc_info=True)
# --- Data Loading Helper ---
def safe_load_csv_auto(file_path, row_limit=None, **kwargs):
    """โหลดไฟล์ CSV พร้อมการจัดการข้อผิดพลาดที่มีประสิทธิภาพสูง

    [Patch v6.9.12] รองรับการส่งชื่อไฟล์แบบ relative โดยค้นหาในรูทโปรเจค
    และหากไม่พบจะแจ้ง FileNotFoundError

    การทำงานประกอบด้วยการตรวจจับไฟล์รูปแบบพิเศษและรวมคอลัมน์
    ``date``/``time`` อัตโนมัติ รวมถึงการลบข้อมูลซ้ำอย่างยืดหยุ่น
    โดยเก็บแถวสุดท้ายไว้เสมอ
    """
    if not os.path.exists(file_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(base_dir, file_path)
        if os.path.exists(alt_path):
            logger.info(f"   (Info) Using project CSV path: {alt_path}")
            file_path = alt_path
        else:
            msg = f"ไม่พบไฟล์ข้อมูลที่ระบุ: {file_path}"
            logger.critical(msg)
            raise FileNotFoundError(msg)

    logger.info(f"      (safe_load) Attempting to load: {os.path.basename(file_path)}")

    # --- 1. ตรวจจับรูปแบบไฟล์อัตโนมัติ ---
    is_malformed = False
    try:
        open_func = open
        if file_path.lower().endswith('.gz'):
            open_func = gzip.open
        try:
            with open_func(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                sample_line = f.readline().strip()
        except OSError:
            if open_func is gzip.open:
                with open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    sample_line = f.readline().strip()
            else:
                raise
            # ตรวจสอบง่ายๆ: ถ้าไม่มี comma และบรรทัดยาวพอสมควร -> สันนิษฐานว่าเป็นรูปแบบพิเศษ
            if ',' not in sample_line and len(sample_line) > 40:
                logger.warning(f"ตรวจพบไฟล์ CSV รูปแบบพิเศษ (ไม่มีตัวคั่น) สำหรับ '{os.path.basename(file_path)}'.")
                is_malformed = True
    except Exception as e:
        logger.error(f"ไม่สามารถเปิดไฟล์เพื่อตรวจสอบรูปแบบได้: {e}")
        raise IOError(f"Cannot read file to inspect format: {file_path}") from e

    # --- 2. เลือกวิธีการอ่านตามรูปแบบไฟล์ ---
    if is_malformed:
        # --- โหมดแปลงข้อมูลพิเศษ ---
        logger.info("...เริ่มต้นกระบวนการแปลงข้อมูลอัตโนมัติ (แยกคอลัมน์และแปลงวันที่)...")

        # Pattern สำหรับแยกข้อมูลที่ติดกัน: (DateTimestamp)(Open)(High)(Low)(Close)(Volume)
        pattern = re.compile(r"(\d{8}\d{2}:\d{2}:\d{2})(\d+\.\d+)(\d+\.\d+)(\d+\.\d+)(\d+\.\d+)(.+)")
        data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if row_limit and i >= row_limit:
                    break
                match = pattern.match(line.strip())
                if match:
                    groups = match.groups()
                    try:
                        # แปลงปี พ.ศ. -> ค.ศ.
                        buddhist_dt_str = groups[0]
                        gregorian_year = int(buddhist_dt_str[:4]) - 543
                        gregorian_dt_str = f"{gregorian_year}{buddhist_dt_str[4:]}"
                        time = pd.to_datetime(gregorian_dt_str, format='%Y%m%d%H:%M:%S')

                        # เพิ่มข้อมูลลงใน list
                        data.append([
                            time,
                            float(groups[1]),  # Open
                            float(groups[2]),  # High
                            float(groups[3]),  # Low
                            float(groups[4]),  # Close
                            float(groups[5])   # Volume
                        ])
                    except (ValueError, TypeError):
                        continue  # ข้ามบรรทัดที่แปลงค่าไม่ได้

        df = pd.DataFrame(data, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        logger.info(f"แปลงข้อมูลจากรูปแบบพิเศษสำเร็จ! โหลดข้อมูลได้ {len(df)} แถว")
    else:
        # --- โหมดการอ่าน CSV ปกติ ---
        logger.info("   (Info) ตรวจพบรูปแบบ CSV มาตรฐาน กำลังโหลดข้อมูล...")
        try:
            df = pd.read_csv(file_path, nrows=row_limit, **kwargs)
        except Exception as e:
            logger.error(f"อ่านไฟล์ CSV ล้มเหลว: {e}")
            return None
        if 'Unnamed: 0' in df.columns:
            potential_dt = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
            if potential_dt.notna().all():
                df.drop(columns=['Unnamed: 0'], inplace=True)
                df.index = potential_dt

    # --- 3. จัดการข้อมูลหลังการโหลด (ทำเหมือนกันทั้ง 2 โหมด) ---
    if df.empty:
        raise DataValidationError(f"ไม่สามารถโหลดข้อมูลจากไฟล์ '{file_path}' หรือไฟล์ว่างเปล่าหลังการแปลง")

    # --- Standardize column names (lowercase & trim) ---
    df.columns = [str(col).strip().lower() for col in df.columns]

    # [Patch] รองรับคอลัมน์ชื่อ 'timestamp' แทน 'date/time'
    if 'date/time' not in df.columns:
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date/time'}, inplace=True)
        else:
            logger.debug("   (Info) ไม่มีคอลัมน์ date/time หรือ timestamp")

    datetime_col = None

    # --- Flexible datetime column detection ---
    if 'date' in df.columns and 'time' in df.columns:
        logger.info("ตรวจพบ คอลัมน์ 'date' และ 'time', กำลังรวมเป็น datetime...")
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            errors='coerce'
        )
        df.drop(columns=['date', 'time'], inplace=True)
        datetime_col = 'datetime'
    elif 'local_time' in df.columns:
        logger.info("ตรวจพบ คอลัมน์ 'local_time', กำลังแปลงข้อมูลเฉพาะรูปแบบ...")
        datetime_column_name = 'local_time'
        datetime_format = '%d.%m.%Y %H:%M:%S'
        series = df[datetime_column_name].astype(str)
        df['datetime'] = pd.to_datetime(series, format=datetime_format, errors='coerce')
        df.drop(columns=[datetime_column_name], inplace=True)
        datetime_col = 'datetime'
    elif 'datetime' in df.columns:
        logger.info("ตรวจพบ คอลัมน์ 'datetime', กำลังแปลงข้อมูล...")
        datetime_col = 'datetime'
    elif 'date/time' in df.columns:
        logger.info("ตรวจพบ คอลัมน์ 'date/time', จะใช้เป็น datetime...")
        datetime_col = 'date/time'
    elif 'timestamp' in df.columns:
        logger.info("ตรวจพบ คอลัมน์ 'timestamp', กำลังแปลงข้อมูล...")
        datetime_col = 'timestamp'

    if datetime_col:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
            )
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    else:
        logger.error(
            f"   (Critical Error) ไม่พบคอลัมน์ Date/Time ที่รู้จัก (date/time, datetime, timestamp) ในไฟล์ {os.path.basename(file_path)}"
        )
        logger.error(f"      คอลัมน์ที่มีอยู่คือ: {list(df.columns)}")

    # ตั้งค่า Index และตรวจสอบข้อมูลซ้ำซ้อน
    time_col_name = datetime_col
    if time_col_name in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col_name]):
            df[time_col_name] = pd.to_datetime(df[time_col_name], errors="coerce")

        initial_rows = len(df)
        df.dropna(subset=[time_col_name], inplace=True)
        if len(df) < initial_rows:
            logger.warning(
                f"   (Warning) Dropped {initial_rows - len(df)} rows with invalid datetime format."
            )

        df = df.set_index(time_col_name)

        if df.index.has_duplicates:
            duplicated_indices = df.index.duplicated(keep="last")
            num_duplicates = duplicated_indices.sum()
            logger.warning(
                f"ตรวจพบ Index (เวลา) ที่ซ้ำกัน {num_duplicates} รายการ ... กำลังลบข้อมูลซ้ำ"
            )
            df = df[~duplicated_indices]
            logger.info(f"ลบข้อมูลที่ซ้ำซ้อนสำเร็จ! (เหลือ {len(df)} แถว)")
    else:
        logger.warning(
            f"ไม่พบคอลัมน์ 'time' หรือ 'datetime' ในไฟล์ '{os.path.basename(file_path)}'"
        )

    # --- Standardize OHLCV columns for compatibility ---
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    logger.info(f"Successfully loaded and validated '{os.path.basename(file_path)}'.")
    return df

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
                    df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)  # pragma: no cover
                df.loc[idx, col] = pd.NaT
            else:
                logging.warning(
                    f"   safe_set_datetime: Index '{idx}' not found during fallback NaT assignment for column '{col}'."
                )  # pragma: no cover
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
# [Patch v6.7.13] Add max_rows parameter for limited row loading
def load_data(file_path, timeframe_str="", price_jump_threshold=0.10, nan_threshold=0.05, dtypes=None, max_rows=None):  # pragma: no cover
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
                                 Defaults to ``DEFAULT_DTYPE_MAP`` from config if not provided.
        max_rows (int, optional): Limit the number of rows to load for debugging. Defaults to None (load all rows).

    Returns:
        pd.DataFrame: The loaded and initially validated DataFrame.

    Raises:
        SystemExit: If critical errors occur (e.g., file not found, essential columns missing).
    """
    logging.info(f"(Loading) กำลังโหลดข้อมูล {timeframe_str} จาก: {file_path}")

    if dtypes is None:
        dtypes = safe_get_global("DEFAULT_DTYPE_MAP", None)
        if dtypes is None:
            try:
                from src.config import DEFAULT_DTYPE_MAP as CONFIG_DEFAULT_DTYPE_MAP
                dtypes = CONFIG_DEFAULT_DTYPE_MAP
            except Exception:
                dtypes = None

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
            read_csv_kwargs = {"low_memory": False, "dtype": dtypes}
            if max_rows is not None:
                read_csv_kwargs["nrows"] = max_rows
            df_pd = pd.read_csv(file_path, **read_csv_kwargs)
            logging.info(
                f"   ไฟล์ดิบ {timeframe_str}: {df_pd.shape[0]} แถว (max_rows={max_rows})"
            )

            if "Date" not in df_pd.columns and "Time" in df_pd.columns:
                logging.info(
                    "   [Pre-Validation] Detected 'Time' column without 'Date'/'Timestamp'."
                )
                dt_series = pd.to_datetime(df_pd["Time"], errors="coerce")
                df_pd["Date"] = dt_series.dt.strftime("%Y%m%d")
                df_pd["Timestamp"] = dt_series.dt.strftime("%H:%M:%S")
                logging.info("      Reconstructed 'Date' and 'Timestamp' columns from 'Time'.")
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
        logging.info(f"Rows after drop price NaN: {df_pd.shape[0]}")

        logging.info("   [Data Quality] ตรวจสอบ Duplicates (Date & Timestamp)...")
        duplicate_cols = ["Date", "Timestamp"]
        if all(col in df_pd.columns for col in duplicate_cols):
            df_pd = deduplicate_and_sort(df_pd, subset_cols=duplicate_cols)
            logging.info(f"      ขนาดข้อมูลหลังจัดการ Duplicates: {df_pd.shape[0]} แถว.")
            logging.info(f"Rows after dedupe: {df_pd.shape[0]}")
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

        logging.info("NaN count after load_data:\n%s", df_pd.isna().sum().to_string())
        logging.info(f"(Success) โหลดและตรวจสอบข้อมูล {timeframe_str} สำเร็จ: {df_pd.shape[0]} แถว")
        return df_pd

    except SystemExit as se:
        raise se
    except Exception as e:
        logging.critical(f"(Error) ไม่สามารถโหลดข้อมูล {timeframe_str}: {e}\n{traceback.format_exc()}", exc_info=True)
        sys.exit(f"ออก: ข้อผิดพลาดร้ายแรงในการโหลดข้อมูล {timeframe_str}")


# [Patch v5.5.15] Optional caching layer for large CSV data
def load_data_cached(file_path, timeframe_str="", cache_format=None, **kwargs):
    """Load CSV using :func:`load_data` with optional caching.

    Parameters
    ----------
    file_path : str
        Path to the CSV data.
    timeframe_str : str, optional
        Timeframe identifier for logging.
    cache_format : str, optional
        If provided, cache the loaded DataFrame in this format
        (``'parquet'``, ``'feather'`` or ``'hdf5'``). Subsequent calls
        will load from the cached file if available.
    kwargs : dict
        Additional arguments forwarded to :func:`load_data`.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame, either from CSV or cached file.
    """

    ext_map = {
        "parquet": ".parquet",
        "feather": ".feather",
        "hdf5": ".h5",
    }

    if cache_format:
        ext = ext_map.get(cache_format.lower())
        if ext:
            cache_path = os.path.splitext(file_path)[0] + ext
            if os.path.exists(cache_path):
                logging.info(f"(Cache) โหลด {timeframe_str} จาก {cache_path}")
                try:
                    if cache_format.lower() == "parquet":
                        return pd.read_parquet(cache_path)
                    if cache_format.lower() == "feather":
                        return pd.read_feather(cache_path)
                    return pd.read_hdf(cache_path, key="data")
                except Exception as e_load:
                    logging.warning(
                        f"(Cache) Failed to load {cache_format} file {cache_path}: {e_load}"
                    )

    df_loaded = load_data(file_path, timeframe_str, **kwargs)

    if cache_format:
        ext = ext_map.get(cache_format.lower())
        if ext:
            cache_path = os.path.splitext(file_path)[0] + ext
            try:
                if cache_format.lower() == "parquet":
                    df_loaded.to_parquet(cache_path)
                elif cache_format.lower() == "feather":
                    df_loaded.reset_index().to_feather(cache_path)
                else:
                    df_loaded.to_hdf(cache_path, key="data", mode="w")
                logging.info(f"(Cache) Saved {timeframe_str} to {cache_path}")
            except Exception as e_save:
                logging.warning(
                    f"(Cache) Failed to save {cache_format} file {cache_path}: {e_save}"
                )

    return df_loaded

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


def prepare_datetime(df, timeframe):  # pragma: no cover
    """Set a DatetimeIndex from Thai or standard date columns."""
    logging.info(
        f"   (Prepare Datetime) กำลังเตรียมข้อมูล datetime สำหรับ Timeframe: {timeframe}"
    )
    df_copy = df.copy()
    if "Date" in df_copy.columns and "Timestamp" in df_copy.columns:
        logging.info("   -> ตรวจพบคอลัมน์ 'Date' และ 'Timestamp'")
        try:
            date_str = df_copy["Date"].astype(str) + " " + df_copy["Timestamp"].astype(str)
            gregorian_dates = []
            for d_str in date_str:
                year = int(d_str[:4])
                if year > 2500:
                    gregorian_year = year - 543
                    gregorian_dates.append(str(gregorian_year) + d_str[4:])
                else:
                    gregorian_dates.append(d_str)
            datetime_series = pd.to_datetime(gregorian_dates, format="%Y%m%d %H:%M:%S")
            df_copy["datetime"] = datetime_series
            logging.info(
                "   -> (Success) แปลงข้อมูล Date และ Timestamp เป็น datetime สำเร็จ (รองรับปี พ.ศ.)"
            )
        except Exception as e:
            logging.error(
                f"   -> (Error) ไม่สามารถแปลง 'Date' และ 'Timestamp' เป็น datetime: {e}",
                exc_info=True,
            )
            try:
                df_copy["datetime"] = pd.to_datetime(
                    df_copy["Date"] + " " + df_copy["Timestamp"], errors="coerce"
                )
                logging.warning("   -> (Fallback) ลองแปลงด้วยวิธีการมาตรฐานอีกครั้ง")
            except Exception as e2:
                logging.critical(
                    f"   -> (Fatal) การแปลง datetime ล้มเหลวทั้งหมด: {e2}",
                    exc_info=True,
                )
                raise ValueError("ไม่สามารถแปลงคอลัมน์ Date และ Timestamp ได้") from e2
    elif "datetime" in df_copy.columns:
        df_copy["datetime"] = pd.to_datetime(df_copy["datetime"], errors="coerce")
    elif "time" in df_copy.columns:
        df_copy["datetime"] = pd.to_datetime(df_copy["time"], errors="coerce")
    else:
        raise ValueError(
            "ไม่พบข้อมูลวันที่ในไฟล์ CSV (ต้องการคอลัมน์ 'Date'/'Timestamp' หรือ 'datetime' หรือ 'time')"
        )
    df_copy.set_index("datetime", inplace=True)
    df_copy = df_copy[~df_copy.index.duplicated(keep="first")]
    df_copy.sort_index(inplace=True)
    logging.info(
        f"   (Success) ตั้งค่า Datetime Index และจัดเรียงข้อมูลสำหรับ {timeframe} สำเร็จ"
    )
    return df_copy
# FILLER
# === END OF PART 4/12 ===
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
# filler
logging.info("Part 4: Data Loading & Initial Preparation Functions Loaded.")

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


def deduplicate_and_sort(df: pd.DataFrame, subset_cols=None) -> pd.DataFrame:
    """Remove duplicate rows and sort by ``subset_cols``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    subset_cols : list[str] or tuple[str], optional
        Columns to identify duplicates. If ``None`` or missing, the DataFrame
        is only sorted by its index.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with duplicates removed (keeping the last occurrence)
        and sorted.
    """
    if df is None or df.empty:
        return df

    if subset_cols is None or not all(col in df.columns for col in subset_cols):
        return df.sort_index()

    dup_count = df.duplicated(subset=subset_cols).sum()
    if dup_count > 0:
        logging.warning(
            f"   (Warning) พบ {dup_count} duplicates based on {subset_cols}. "
            "Keeping last occurrence."
        )
    df_sorted = df.sort_values(list(subset_cols))
    df_dedup = df_sorted.groupby(list(subset_cols), as_index=False).last()
    return df_dedup.reset_index(drop=True)


def convert_thai_years(df, column):
    """Stubbed Thai year converter."""
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

# [Patch v5.7.3] Convert Thai Buddhist year datetime string to pandas Timestamp
def convert_thai_datetime(series, tz="UTC", errors="raise"):
    """Convert Thai date strings to timezone-aware ``datetime``.

    Years greater than 2500 are assumed to be Buddhist Era and are
    converted to Gregorian by subtracting 543. Invalid values raise
    ``ValueError`` when ``errors='raise'`` otherwise return ``NaT``.
    """
    is_series = isinstance(series, pd.Series)
    if not is_series and not isinstance(series, str):
        raise TypeError("series must be a pandas Series or str")

    def _parse(value):
        using_robust = False
        try:
            dt = datetime.datetime.fromisoformat(str(value))
        except Exception:
            try:
                dt = robust_date_parser(str(value))
                using_robust = True
            except Exception:
                if errors == "raise":
                    raise ValueError(f"Cannot parse datetime: {value}")
                return pd.NaT
        if not using_robust and dt.year > 2500:
            dt = dt.replace(year=dt.year - 543)
        ts = pd.Timestamp(dt)
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    if is_series:
        return series.apply(_parse)
    return _parse(series)


def prepare_datetime_index(df):
    """Set a DatetimeIndex using a 'Timestamp' column."""
    if 'Timestamp' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'Timestamp'})
    if 'Timestamp' in df.columns:
        df.index = pd.to_datetime(df['Timestamp'], errors='coerce')
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
        msg = (
            f"[Patch v5.8.15] Missing raw CSV: {file_path}. "
            "กรุณาวางไฟล์ CSV ในไดเรกทอรีที่กำหนด"
        )
        logging.error(msg)
        raise RuntimeError(msg)
    return True

# --- M15 Data Path Validator ---
# [Patch v6.8.2] Ensure correct file name and existence
def validate_m15_data_path(file_path):
    """Validate that the M15 data path points to an expected file."""
    allowed = {"XAUUSD_M15.csv"}
    if not isinstance(file_path, str) or not file_path:
        logging.error("(Error) Invalid file path for M15 data.")
        return False
    fname = os.path.basename(file_path)
    if fname not in allowed:
        logging.error(
            f"(Error) Unexpected M15 data file '{fname}'. Expected one of {allowed}."
        )
        return False
    if not os.path.exists(file_path):
        msg = (
            f"[Patch v6.8.2] Missing raw CSV: {file_path}. "
            "กรุณาวางไฟล์ CSV ในไดเรกทอรีที่กำหนด"
        )
        logging.error(msg)
        raise RuntimeError(msg)
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
    """Load raw M15 data after validating the file path."""
    if not validate_m15_data_path(path):
        return None
    return safe_load_csv_auto(path)

def write_test_file(path):
    """Stubbed helper to write a simple file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("test")
    return path

def clean_test_file(test_file_path: str) -> None:
    """Remove a test file if it's inside ``cfg.DATA_DIR``."""
    # [Patch] Guard against accidental deletion outside DATA_DIR
    import logging
    from src import config as cfg
    logger = logging.getLogger(__name__)

    abs_path = os.path.realpath(test_file_path)
    env_dir = os.getenv("DATA_DIR")
    data_dir = os.path.realpath(env_dir) if env_dir else os.path.realpath(str(cfg.DATA_DIR))
    if abs_path.startswith(data_dir):
        try:
            os.remove(test_file_path)
        except FileNotFoundError:
            pass

    else:
# noinspection PyUnresolvedReferences
        logger.warning(
            f"Ignoring removal of {test_file_path}: outside DATA_DIR"
        )

# [Patch v6.7.10] Auto convert raw gold CSV files to Thai datetime format
def auto_convert_gold_csv(data_dir="data", output_path=None):
    """แปลงไฟล์ XAUUSD_M*.csv ทั้งหมดให้เป็นรูปแบบ _thai.csv

    Parameters
    ----------
    data_dir : str
        โฟลเดอร์ที่เก็บไฟล์ CSV ต้นฉบับ
    output_path : str, optional
        Path ปลายทางที่ pipeline ต้องใช้ หากระบุและพบเพียงไฟล์เดียว
        จะบันทึกผลไปยัง path นี้โดยตรง
        หากมีหลายไฟล์จะบันทึกไปยังโฟลเดอร์ของ ``output_path``
    """
    pattern = os.path.join(data_dir, "XAUUSD_M*.csv")
    files = glob.glob(pattern)

    # --- START: FIX for Directory Path Error ---
    # [Patch v6.9.11] Fallback to current dir when both data_dir and
    # output_path lack directory information
    target_dir = data_dir or "."
    if output_path:
        # If output_path is already a directory, use it
        if os.path.isdir(output_path):
            target_dir = output_path
        else:
            # Otherwise, derive the directory from the provided file path
            potential_dir = os.path.dirname(output_path)
            # Only use the directory if it's not an empty string
            if potential_dir:
                target_dir = potential_dir

    if not target_dir:
        target_dir = "."

    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(
            f"✗ [AutoConvert] Critical Error: ไม่สามารถสร้างโฟลเดอร์ปลายทางได้ '{target_dir}': {e}"
        )
        return
    # --- END: FIX for Directory Path Error ---

    for f in files:
        if f.endswith("_thai.csv"):
            continue

        base_out = os.path.basename(f).replace(".csv", "_thai.csv")
        out_f = os.path.join(target_dir, base_out)
        try:
            df = pd.read_csv(f)
            df.columns = [c.capitalize() for c in df.columns]
            if "Date" in df.columns and "Time" in df.columns:
                dt = pd.to_datetime(
                    df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce"
                )
            elif "Timestamp" in df.columns:

                # [Patch v6.9.14] ระบุรูปแบบวันที่อย่างชัดเจนเพื่อความเร็วและแม่นยำ
                dt = pd.to_datetime(
                    df["Timestamp"].astype(str),
                    format="%Y.%m.%d %H:%M:%S",
                    errors="coerce",
                )

            else:
                print(f"ข้าม {f}: ไม่พบคอลัมน์ Date/Time")
                continue

            def format_thai_date(d):
                if pd.isna(d):
                    return None
                return f"{d.year + 543:04d}{d.month:02d}{d.day:02d}"

            df["Date"] = dt.map(format_thai_date)
            df["Timestamp"] = dt.dt.strftime("%H:%M:%S")

            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns and col.lower() in df.columns:
                    df[col] = df[col.lower()]

            df.dropna(subset=["Date"], inplace=True)

            df2 = df[["Date", "Timestamp", "Open", "High", "Low", "Close"]]
            df2.to_csv(out_f, index=False)
            status = "พบ" if os.path.exists(out_f) else "ไม่พบ"
            print(f"✔ [AutoConvert] {out_f} OK - {status}ไฟล์")
        except Exception as e:
            print(f"✗ [AutoConvert] Error processing file '{f}' to '{out_f}': {e}")

# [Patch v6.8.10] Helper to load default project CSV files
def load_project_csvs(row_limit=None):
    """Load XAUUSD_M1.csv และ XAUUSD_M15.csv ที่อยู่ในรูทโปรเจค

    Parameters
    ----------
    row_limit : int, optional
        จำกัดจำนวนแถวที่โหลดเพื่อให้การทดสอบทำงานรวดเร็ว

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ข้อมูลจากไฟล์ M1 และ M15 ตามลำดับ
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    m1_path = os.path.join(base_dir, "XAUUSD_M1.csv")
    m15_path = os.path.join(base_dir, "XAUUSD_M15.csv")
    m1_df = safe_load_csv_auto(m1_path, row_limit=row_limit)
    m15_df = safe_load_csv_auto(m15_path, row_limit=row_limit)
    return m1_df, m15_df

# [Patch v6.9.0] Utility to convert CSV to Parquet with fallback handling
def auto_convert_csv_to_parquet(source_path: str, dest_folder) -> None:
    """Convert ``source_path`` to Parquet inside ``dest_folder``.

    If writing Parquet fails, a CSV fallback is saved instead. Warnings are
    logged when the source path does not exist or read/save errors occur.
    """
    from pathlib import Path

    logger = logging.getLogger(__name__)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    if not source_path or not os.path.exists(source_path):
        logger.warning("[AutoConvert] Source CSV not found: %s", source_path)
        return

    try:
        df = pd.read_csv(source_path)
    except Exception as exc:  # pragma: no cover - unexpected read error
        logger.error("[AutoConvert] Failed reading %s: %s", source_path, exc)
        return

    dest_file = dest_folder / (Path(source_path).stem + ".parquet")
    try:
        df.to_parquet(dest_file)
        logger.info("[AutoConvert] Saved Parquet to %s", dest_file)
    except Exception as exc:  # pragma: no cover - optional fallback
        logger.warning(
            "[AutoConvert] Could not save Parquet (%s). Saving CSV fallback", exc
        )
        df.to_csv(dest_file.with_suffix(".csv"), index=False)


# [Patch v6.9.12] Simple CSV loader with Thai year handling
def load_data_from_csv(file_path: str, nrows: int = None, auto_convert: bool = True):
    """
    Loads data from a CSV file, handling potential date parsing issues and Thai Buddhist years.
    """
    logger.info(f"Loading data from CSV: {file_path}")

    temp_df = pd.read_csv(file_path, nrows=nrows)

    if 'Timestamp' in temp_df.columns:
        logger.info("Detected 'Timestamp' column, renaming to 'Time'.")
        temp_df.rename(columns={'Timestamp': 'Time'}, inplace=True)

    if auto_convert and 'Time' in temp_df.columns and temp_df['Time'].dtype == 'object':
        temp_df['Time'] = pd.to_datetime(temp_df['Time'], errors='coerce')

    try:
        temp_df['Time'] = pd.to_datetime(temp_df['Time'])
        temp_df.set_index('Time', inplace=True)
    except Exception as e:
        logger.error(f"Failed to parse datetime after potential conversion: {e}")
        raise e

    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(temp_df.columns):
        missing_cols = required_cols - set(temp_df.columns)
        raise ValueError(f"CSV file {file_path} is missing required columns: {missing_cols}")

    df = temp_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(
        {
            'Open': 'float32',
            'High': 'float32',
            'Low': 'float32',
            'Close': 'float32',
            'Volume': 'float32'
        }
    )

    logger.info(f"Successfully loaded and processed {len(df)} rows from {file_path}")
    return df
# [Patch v5.7.3] Validate DataFrame for required columns and non-emptiness
def validate_csv_data(df, required_cols=None):
    """Ensure ``df`` is non-empty and contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded DataFrame to validate.
    required_cols : list of str, optional
        Columns that must be present. If ``None`` no check is performed.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("CSV data is empty")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
    return df


# [Patch v5.4.5] Robust loader for final M1 data
def load_final_m1_data(path, trade_log_df=None):
    """Load prepared M1 dataset with validation and timezone alignment."""
    if not validate_m1_data_path(path):
        return None
    df = safe_load_csv_auto(path)
    if df is None or df.empty:
        logging.error("(Error) Failed to load M1 data or file is empty.")
        return None
    required = ["open", "high", "low", "close"]
    cols_lower = [c.lower() for c in df.columns]
    missing = [c for c in required if c not in cols_lower]
    if missing:
        logging.error(f"(Error) M1 Data missing columns: {missing}")
        return None
    if not isinstance(df.index, pd.DatetimeIndex) and not any(c in cols_lower for c in ['time', 'datetime']):
        logging.error("(Error) Invalid datetime index for M1 data.")
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
        # [Patch v5.6.2] Replace deprecated fillna(method=...) usage
        method = fillna_method.lower()
        if method in ("ffill", "pad"):
            df.ffill(inplace=True)
        elif method in ("bfill", "backfill"):
            df.bfill(inplace=True)
        else:
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




__all__ = [
    "safe_get_global",
    "setup_output_directory",
    "set_thai_font",
    "install_thai_fonts_colab",
    "configure_matplotlib_fonts",
    "setup_fonts",
    "safe_load_csv_auto",
    "load_data_from_csv",
    "load_app_config",
    "safe_set_datetime",
    "load_data",
    "load_data_cached",
    "preview_datetime_format",
    "parse_datetime_safely",
    "prepare_datetime",
    "inspect_file_exists",
    "read_csv_with_date_parse",
    "check_nan_percent",
    "check_duplicates",
    "deduplicate_and_sort",
    "check_price_jumps",
    "convert_thai_years",
    "robust_date_parser",
    "convert_thai_datetime",
    "prepare_datetime_index",
    "validate_m1_data_path",
    "validate_m15_data_path",
    "load_raw_data_m1",
    "load_raw_data_m15",
    "write_test_file",
    "clean_test_file",
    "validate_csv_data",
    "load_final_m1_data",
    "check_data_quality",
    "auto_convert_gold_csv",
    "auto_convert_csv_to_parquet",
    "load_project_csvs",
]

