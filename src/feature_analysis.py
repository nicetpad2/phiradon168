import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import features as feat
from src.config import LOG_DIR

logger = logging.getLogger(__name__)


def analyze_feature_distribution(
    df: pd.DataFrame,
    feature_list: List[str],
    output_dir: str = os.path.join(LOG_DIR, "feature_analysis"),
) -> Optional[Dict[str, Dict[str, float]]]:
    """Calculate summary statistics and histogram data for a list of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่ใช้วิเคราะห์ฟีเจอร์
    feature_list : List[str]
        รายชื่อคอลัมน์ที่จะวิเคราะห์
    output_dir : str, optional
        โฟลเดอร์สำหรับบันทึกรูปภาพ

    Returns
    -------
    Optional[Dict[str, Dict[str, float]]]
        ค่าเฉลี่ย ค่ามัธยฐาน ส่วนเบี่ยงเบนมาตรฐาน และข้อมูล histogram ต่อฟีเจอร์
    """
    os.makedirs(output_dir, exist_ok=True)
    stats: Dict[str, Dict[str, float]] = {}
    for feature in feature_list:
        if feature not in df.columns:
            logger.error("Feature %s not found in DataFrame", feature)
            return None
        series = pd.to_numeric(df[feature], errors="coerce")
        counts, bins = np.histogram(series.dropna(), bins=50)
        stats[feature] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "hist_bins": bins.tolist(),
            "hist_counts": counts.tolist(),
        }
        plt.figure()
        series.plot.hist(bins=50)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.savefig(os.path.join(output_dir, f"{feature}_hist.png"))
        plt.close()

        plt.figure()
        series.plot.box()
        plt.title(f"Boxplot of {feature}")
        plt.savefig(os.path.join(output_dir, f"{feature}_box.png"))
        plt.close()
    return stats


def detect_low_variance_features(
    df: pd.DataFrame,
    feature_list: List[str],
    threshold: float = 1e-6,
) -> List[str]:
    """Identify features with near-zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่ใช้ตรวจสอบ
    feature_list : List[str]
        รายชื่อคอลัมน์ที่จะตรวจสอบ
    threshold : float, optional
        ค่า std ต่ำสุดที่ยอมรับได้

    Returns
    -------
    List[str]
        รายชื่อฟีเจอร์ที่มี variance ต่ำหรือมีค่าเดียว
    """
    low_var = []
    for feature in feature_list:
        if feature not in df.columns:
            logger.error("Feature %s not found in DataFrame", feature)
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        if series.nunique(dropna=False) <= 1 or series.std() <= threshold:
            low_var.append(feature)
    return low_var


def select_top_pnl_features(
    df: pd.DataFrame,
    target_col: str = "pnl_usd_net",
    n: int = 10,
) -> List[str]:
    """Return top features most correlated with target PnL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่ใช้คำนวณ correlation
    target_col : str, optional
        ชื่อคอลัมน์กำไรขาดทุน
    n : int, optional
        จำนวนฟีเจอร์ที่ต้องการ

    Returns
    -------
    List[str]
        รายชื่อฟีเจอร์ที่มีความสัมพันธ์สูงสุดกับ target
    """
    if target_col not in df.columns:
        logger.error("Target column %s not found in DataFrame", target_col)
        return []
    numeric_df = df.select_dtypes(include=[float, int])
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    corr = numeric_df.corr()[target_col].drop(target_col)
    corr = corr.abs().sort_values(ascending=False)
    return corr.head(n).index.tolist()


def calculate_correlation_matrix(
    df: pd.DataFrame,
    feature_list: List[str],
    output_dir: str = os.path.join(LOG_DIR, "feature_analysis"),
) -> Optional[pd.DataFrame]:
    """คำนวณ correlation matrix สำหรับฟีเจอร์ที่ระบุ

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่ใช้คำนวณ
    feature_list : List[str]
        รายชื่อคอลัมน์ที่จะคำนวณ
    output_dir : str, optional
        โฟลเดอร์สำหรับบันทึกผลลัพธ์

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame correlation matrix หรือ None หากมีคอลัมน์หาย
    """
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        logger.error("Features %s not found in DataFrame", missing)
        return None
    numeric_df = df[feature_list].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr()
    os.makedirs(output_dir, exist_ok=True)
    corr_path = os.path.join(output_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    return corr_matrix


def compare_in_out_distribution(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    feature_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """เปรียบเทียบสถิติเบื้องต้นของฟีเจอร์ระหว่างชุด In-Sample และ Out-of-Sample.

    Parameters
    ----------
    df_is : pd.DataFrame
        DataFrame ของข้อมูล In-Sample
    df_oos : pd.DataFrame
        DataFrame ของข้อมูล Out-of-Sample
    feature_list : List[str]
        รายชื่อฟีเจอร์ที่ต้องการเปรียบเทียบ

    Returns
    -------
    Dict[str, Dict[str, float]]
        ค่าสถิติ mean, std และ skewness ของแต่ละชุดข้อมูล
    """

    stats: Dict[str, Dict[str, float]] = {}
    for feature in feature_list:
        if feature not in df_is.columns or feature not in df_oos.columns:
            logger.error("Feature %s not found in both DataFrames", feature)
            continue
        series_is = pd.to_numeric(df_is[feature], errors="coerce")
        series_oos = pd.to_numeric(df_oos[feature], errors="coerce")
        stats[feature] = {
            "mean_is": float(series_is.mean()),
            "mean_oos": float(series_oos.mean()),
            "std_is": float(series_is.std()),
            "std_oos": float(series_oos.std()),
            "skew_is": float(series_is.skew()),
            "skew_oos": float(series_oos.skew()),
        }
    return stats


def main(sample_rows: int = 5000):  # pragma: no cover - CLI helper
    """Run feature distribution analysis on a sample of the M1 dataset.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], List[str], pd.DataFrame]
        สถิติฟีเจอร์, รายชื่อฟีเจอร์ variance ต่ำ และ correlation matrix
    """
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "XAUUSD_M1.csv"
    )
    from src.utils.data_utils import safe_read_csv

    df = safe_read_csv(data_path).head(sample_rows)
    # [Patch v6.9.4] Auto-detect datetime column names for robustness
    possible_time_cols = ["Date", "Date/Time", "Timestamp", "datetime", "Datetime"]
    if {"Date", "Timestamp"}.issubset(df.columns):
        df.index = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Timestamp"], errors="coerce"
        )
    else:
        time_col = next((c for c in df.columns if c in possible_time_cols), None)
        if time_col is None and "Time" in df.columns:
            time_col = "Time"
        if time_col is None:
            logger.error("Missing Date/Timestamp columns in dataset")
            return {}, [], pd.DataFrame()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
        df.index = df[time_col]
    df_feat = feat.engineer_m1_features(df)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reports", "feature_analysis", timestamp_str)

    features_to_analyze = [
        "spike_score",
        "cluster",
        "RSI",
        "MACD_line",
        "ATR_14_Rolling_Avg",
    ]
    stats = analyze_feature_distribution(
        df_feat, features_to_analyze, output_dir=output_dir
    )
    low_var = detect_low_variance_features(df_feat, features_to_analyze)
    corr = calculate_correlation_matrix(
        df_feat, features_to_analyze, output_dir=output_dir
    )

    logger.info("Feature statistics: %s", stats)
    logger.info("Low variance features: %s", low_var)

    return stats, low_var, corr


if __name__ == "__main__":  # pragma: no cover - CLI entry
    logging.basicConfig(level=logging.INFO)
    main()
