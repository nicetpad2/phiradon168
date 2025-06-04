import os
import logging
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt

from src import features as feat

logger = logging.getLogger(__name__)


def analyze_feature_distribution(df: pd.DataFrame, feature_list: List[str], output_dir: str = "logs/feature_analysis") -> Dict[str, Dict[str, float]]:
    """Generate basic statistics and plots for given features."""
    os.makedirs(output_dir, exist_ok=True)
    stats: Dict[str, Dict[str, float]] = {}
    for feature in feature_list:
        if feature not in df.columns:
            logger.warning("Feature %s not found", feature)
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        stats[feature] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
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


def detect_low_variance_features(df: pd.DataFrame, feature_list: List[str], threshold: float = 1e-6) -> List[str]:
    """Return features with near-zero variance or only one unique value."""
    low_var = []
    for feature in feature_list:
        if feature not in df.columns:
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        if series.nunique(dropna=False) <= 1 or series.std() <= threshold:
            low_var.append(feature)
    return low_var


def main(sample_rows: int = 5000):
    """Run feature distribution analysis on a sample of the M1 dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "XAUUSD_M1.csv")
    df = pd.read_csv(data_path, nrows=sample_rows)
    df.index = pd.to_datetime(df["Date"].astype(str) + " " + df["Timestamp"], errors="coerce")
    df_feat = feat.engineer_m1_features(df)

    features_to_analyze = ["spike_score", "cluster", "RSI", "MACD_line", "ATR_14_Rolling_Avg"]
    stats = analyze_feature_distribution(df_feat, features_to_analyze)
    low_var = detect_low_variance_features(df_feat, features_to_analyze)

    logger.info("Feature statistics: %s", stats)
    logger.info("Low variance features: %s", low_var)

    return stats, low_var


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
