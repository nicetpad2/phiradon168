"""Model-related helpers extracted from src.main"""
import json
import logging
import os
import pandas as pd
from src.utils import download_model_if_missing, download_feature_list_if_missing, validate_file
from src.data_loader import safe_load_csv_auto

META_CLASSIFIER_PATH = 'meta_classifier.pkl'
SPIKE_MODEL_PATH = 'meta_classifier_spike.pkl'
CLUSTER_MODEL_PATH = 'meta_classifier_cluster.pkl'
DEFAULT_MODEL_TO_LINK = 'catboost'
shap_importance_threshold = 0.01
permutation_importance_threshold = 0.001
sample_size = None
features_to_drop = None
early_stopping_rounds_config = 200


def ensure_main_features_file(output_dir):
    """Create default features_main.json if it does not exist."""
    path = os.path.join(output_dir, "features_main.json")
    if os.path.exists(path):
        return path
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    logging.info("[Patch] Created default features_main.json")
    return path


def save_features_main_json(features, output_dir):
    """Save main features list, creating QA log if empty."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "features_main.json")
    if not features:
        logging.warning("[QA] features_main.json is empty. Creating empty features file.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        qa_log = os.path.join(output_dir, "features_main_qa.log")
        with open(qa_log, "w", encoding="utf-8") as f:
            f.write("[QA] features_main.json EMPTY. Please check feature engineering logic.\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        logging.info("[QA] features_main.json saved successfully (%d features).", len(features))
    return path


def save_features_json(features, model_name, output_dir):
    """Save feature list for a specific model name."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"features_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(features if features is not None else [], f, ensure_ascii=False, indent=2)
    return path


def ensure_model_files_exist(output_dir, base_trade_log_path, base_m1_data_path):
    """Ensure all model and feature files exist or auto-train."""
    logging.info("\n--- (Auto-Train Check) Ensuring Model Files Exist ---")
    skip_auto_train = os.getenv("SKIP_AUTO_TRAIN", "0") in {"1", "True", "true"}
    required = {
        "main": (META_CLASSIFIER_PATH, "features_main.json"),
        "spike": (SPIKE_MODEL_PATH, "features_spike.json"),
        "cluster": (CLUSTER_MODEL_PATH, "features_cluster.json"),
    }
    missing_models = []
    for key, (model_file, feature_file) in required.items():
        model_path = os.path.join(output_dir, model_file)
        feature_path = os.path.join(output_dir, feature_file)
        if not (os.path.exists(model_path) and os.path.exists(feature_path)):
            download_model_if_missing(model_path, f"URL_MODEL_{key.upper()}")
            download_feature_list_if_missing(feature_path, f"URL_FEATURES_{key.upper()}")
            if not os.path.exists(model_path) or not os.path.exists(feature_path):
                missing_models.append(key)
                logging.warning("Missing model file for '%s' (%s).", key, model_file)
    if not missing_models:
        logging.info("   (Success) Model files and feature lists already exist.")
        return
    if skip_auto_train:
        logging.warning("   SKIP_AUTO_TRAIN enabled - creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return
    train_log_path = None
    for ext in (".csv.gz", ".csv"):
        candidate = base_trade_log_path + ext
        if os.path.exists(candidate):
            train_log_path = candidate
            break
    m1_path = None
    for ext in (".csv.gz", ".csv"):
        candidate = base_m1_data_path + ext
        if os.path.exists(candidate):
            m1_path = candidate
            break
    if train_log_path is None or m1_path is None:
        logging.error("   (Error) Training data missing. Creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return
    try:
        trade_log_df = safe_load_csv_auto(train_log_path)
    except Exception as e:
        logging.error("   (Error) Failed safe_load_csv_auto: %s", e, exc_info=True)
        try:
            trade_log_df = pd.read_csv(train_log_path)
        except Exception as e2:
            logging.error("   (Error) Fallback read_csv failed: %s", e2, exc_info=True)
            trade_log_df = None
    if trade_log_df is None or trade_log_df.empty:
        logging.error("   (Error) Loaded trade log is empty. Creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return

    # [Patch v6.9.38] fill missing exit_reason column for auto-train stub
    if 'exit_reason' not in trade_log_df.columns:
        trade_log_df['exit_reason'] = 'TP'

    for key in missing_models:
        try:
            from importlib import import_module
            main_mod = import_module('src.main')
            train_and_export_meta_model = getattr(main_mod, 'train_and_export_meta_model')
            saved_paths, features = train_and_export_meta_model(
                trade_log_path=None,
                m1_data_path=m1_path,
                output_dir=output_dir,
                model_purpose=key,
                trade_log_df_override=trade_log_df,
                model_type_to_train="catboost",
                link_model_as_default=DEFAULT_MODEL_TO_LINK,
                enable_dynamic_feature_selection=True,
                feature_selection_method="shap",
                shap_importance_threshold=shap_importance_threshold,
                permutation_importance_threshold=permutation_importance_threshold,
                enable_optuna_tuning=False,
                sample_size=sample_size,
                features_to_drop_before_train=features_to_drop,
                early_stopping_rounds=early_stopping_rounds_config,
            )
            if saved_paths is None or key not in saved_paths:
                raise RuntimeError("Training did not produce a model file")
        except Exception as e:
            logging.error("   (Error) Auto-training failed for '%s': %s", key, e, exc_info=True)
            os.makedirs(output_dir, exist_ok=True)
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
            continue
        model_path = os.path.join(output_dir, required[key][0])
        features_path = os.path.join(output_dir, required[key][1])
        if not os.path.exists(model_path):
            os.makedirs(output_dir, exist_ok=True)
            open(model_path, "a").close()
        if features is None:
            open(features_path, "a").close()
        else:
            if key == "main":
                save_features_main_json(features, output_dir)
            else:
                save_features_json(features, key, output_dir)
        if not validate_file(model_path):
            logging.warning("[QA] Placeholder created for '%s' model", key)
            open(model_path, "a").close()
        if not validate_file(features_path):
            logging.warning("[QA] Placeholder created for '%s' features", key)
            open(features_path, "a").close()
    logging.info("--- (Auto-Train Check) Finished ---")
