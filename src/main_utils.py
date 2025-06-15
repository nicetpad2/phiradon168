import os
import logging
import pandas as pd
from src.data_loader import load_data
from src.csv_validator import validate_and_convert_csv


def ensure_default_output_dir(path):
    """สร้างโฟลเดอร์ผลลัพธ์เริ่มต้นหากยังไม่มี"""
    if not os.path.isabs(path):
        project_root = os.getcwd()
        path = os.path.join(project_root, path)
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"   (Setup) ตรวจสอบโฟลเดอร์ผลลัพธ์: {path}")
        return path
    except Exception as e:  # pragma: no cover - unexpected file errors
        logging.error(f"   (Error) สร้างโฟลเดอร์ผลลัพธ์ไม่สำเร็จ: {e}", exc_info=True)
        return None


def load_validated_csv(raw_path, timeframe, dtypes=None):
    """Validate and load CSV or Parquet, ensuring Buddhist year conversion."""
    if raw_path.endswith(".parquet"):
        try:
            return pd.read_parquet(raw_path)
        except Exception as e_read:
            logging.warning(f"(Warning) Failed to read parquet {raw_path}: {e_read}")
            return load_data(raw_path.replace(".parquet", ".csv"), timeframe, dtypes=dtypes)

    parquet_path = raw_path.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e_read_cache:
            logging.warning(f"(Warning) Failed to read parquet {parquet_path}: {e_read_cache}")

    clean_path = raw_path.replace(".csv", "_clean.csv")
    if not os.path.exists(clean_path):
        print(f"ไฟล์ข้อมูลสะอาด '{clean_path}' ยังไม่มี, กำลังสร้างจากไฟล์ต้นฉบับ...")
        try:
            validate_and_convert_csv(raw_path, clean_path)
            print("สร้างไฟล์ข้อมูลสะอาดสำเร็จ")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดร้ายแรงระหว่างการตรวจสอบและแปลงไฟล์ CSV: {e}")
            raise

    df_loaded = load_data(clean_path, timeframe, dtypes=dtypes)
    try:
        df_loaded.to_parquet(parquet_path)
    except Exception as e_save:
        logging.warning(f"(Warning) Failed to save parquet to {parquet_path}: {e_save}")
    if df_loaded.empty:
        logging.warning("(Warning) Loaded DataFrame is empty after CSV load")
    return df_loaded
