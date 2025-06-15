from .common import *
from .engineering import DEFAULT_META_CLASSIFIER_FEATURES
import os
import json
import matplotlib.pyplot as plt
import traceback
try:
    import shap
except ImportError:
    shap = None
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None


# --- SHAP Feature Selection Helper Function ---
# [Patch v5.0.2] Exclude SHAP helper from coverage
def select_top_shap_features(shap_values_val, feature_names, shap_threshold=0.01):  # pragma: no cover
    """
    Selects features based on Normalized Mean Absolute SHAP values exceeding a threshold.
    (v4.8.8 Patch 4: Corrected return for invalid feature_names)
    """
    logging.info(f"   [SHAP Select] กำลังเลือก Features ที่มี Normalized SHAP >= {shap_threshold:.4f}...")
    # <<< [Patch] MODIFIED v4.8.8 (Patch 1): Enhanced input validation >>>
    if shap_values_val is None or not isinstance(shap_values_val, np.ndarray) or shap_values_val.size == 0:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: ค่า SHAP ไม่ถูกต้องหรือว่างเปล่า. คืนค่า Features เดิม.")
        return feature_names if isinstance(feature_names, list) else [] # Return original or empty list
    # <<< [Patch] MODIFIED v4.8.8 (Patch 4): Return None if feature_names is invalid >>>
    if feature_names is None or not isinstance(feature_names, list) or not feature_names:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: รายชื่อ Features ไม่ถูกต้องหรือว่างเปล่า. คืนค่า None.")
        return None
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 4) >>>
    if shap_values_val.ndim != 2:
        # Handle potential case where SHAP returns values for multiple classes (e.g., list of arrays)
        if isinstance(shap_values_val, list) and len(shap_values_val) >= 2 and isinstance(shap_values_val[1], np.ndarray) and shap_values_val[1].ndim == 2:
            logging.debug("      (Info) SHAP values appear to be for multiple classes, using index 1 (positive class).")
            shap_values_val = shap_values_val[1] # Use SHAP values for the positive class
        else:
            logging.warning(f"      (Warning) ขนาด SHAP values ไม่ถูกต้อง ({shap_values_val.ndim} dimensions, expected 2). คืนค่า Features เดิม.")
            return feature_names
    if shap_values_val.shape[1] != len(feature_names):
        logging.warning(f"      (Warning) ขนาด SHAP values ไม่ตรงกับจำนวน Features (SHAP: {shap_values_val.shape[1]}, Features: {len(feature_names)}). คืนค่า Features เดิม.")
        return feature_names
    if shap_values_val.shape[0] == 0:
        logging.warning("      (Warning) SHAP values array มี 0 samples. ไม่สามารถคำนวณ Importance ได้. คืนค่า Features เดิม.")
        return feature_names
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 1) >>>

    try:
        mean_abs_shap = np.abs(shap_values_val).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) พบ NaN หรือ Inf ใน Mean Absolute SHAP values. ไม่สามารถเลือก Features ได้. คืนค่า Features เดิม.")
            return feature_names

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        if total_shap > 1e-9:
            shap_df["Normalized_SHAP"] = shap_df["Mean_Abs_SHAP"] / total_shap
        else:
            shap_df["Normalized_SHAP"] = 0.0
            logging.warning("      (Warning) Total Mean Abs SHAP ใกล้ศูนย์, ไม่สามารถ Normalize ได้. จะไม่เลือก Feature ใดๆ.")
            return [] # Return empty list if no importance

        selected_features_df = shap_df[shap_df["Normalized_SHAP"] >= shap_threshold].copy()
        selected_features = selected_features_df["Feature"].tolist()

        if not selected_features:
            logging.warning(f"      (Warning) ไม่มี Features ใดผ่านเกณฑ์ SHAP >= {shap_threshold:.4f}. คืนค่า List ว่าง.")
            return []
        elif len(selected_features) < len(feature_names):
            removed_features = sorted(list(set(feature_names) - set(selected_features)))
            logging.info(f"      (Success) เลือก Features ได้ {len(selected_features)} ตัวจาก SHAP.")
            logging.info(f"         Features ที่ถูกตัดออก {len(removed_features)} ตัว: {removed_features}")
            logging.info("         Features ที่เลือก (เรียงตามค่า SHAP):")
            logging.info("\n" + selected_features_df.sort_values("Normalized_SHAP", ascending=False)[["Feature", "Normalized_SHAP"]].round(5).to_string(index=False))
        else:
            logging.info("      (Success) Features ทั้งหมดผ่านเกณฑ์ SHAP.")
        return selected_features
    except Exception as e:
        logging.error(f"      (Error) เกิดข้อผิดพลาดระหว่างการเลือก Features ด้วย SHAP: {e}. คืนค่า Features เดิม.", exc_info=True)
        return feature_names

# --- Model Quality Check Functions ---
# [Patch v5.0.2] Exclude model overfit check from coverage
def check_model_overfit(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, metric="AUC", threshold_pct=15.0):  # pragma: no cover
    """
    Checks for potential overfitting by comparing model performance.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info(f"   [Check] Checking for Overfitting ({metric})...")
    if model is None:
        logging.warning("      (Warning) Cannot check Overfitting: Model is None.")
        return
    if X_val is None or y_val is None:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data missing.")
        return
    if (X_test is None and y_test is not None) or (X_test is not None and y_test is None):
        logging.warning("      (Warning) Cannot check Overfitting: Test data requires both X_test and y_test.")
        X_test, y_test = None, None

    def _ensure_pd(data, name):
        if data is None: return None
        if isinstance(data, np.ndarray):
            if data.ndim == 1: return pd.Series(data, name=name)
            elif data.ndim == 2:
                feature_names = getattr(model, 'feature_names_', None)
                if feature_names and len(feature_names) == data.shape[1]:
                    return pd.DataFrame(data, columns=feature_names)
                else:
                    return pd.DataFrame(data)
            else:
                logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected dimensions ({data.ndim}).")
                return None
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        else:
            logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected type ({type(data)}).")
            return None

    X_train = _ensure_pd(X_train, "X_train")
    y_train = _ensure_pd(y_train, "y_train")
    X_val = _ensure_pd(X_val, "X_val")
    y_val = _ensure_pd(y_val, "y_val")
    X_test = _ensure_pd(X_test, "X_test")
    y_test = _ensure_pd(y_test, "y_test")

    if X_val is None or y_val is None: return
    if X_train is not None and (y_train is None or len(X_train) != len(y_train)):
        logging.warning("      (Warning) Cannot check Overfitting: Train X and y data sizes do not match or y_train is None.")
        X_train, y_train = None, None
    if X_val is not None and (y_val is None or len(X_val) != len(y_val)):
        logging.warning("      (Warning) Cannot check Overfitting: Validation X and y data sizes do not match or y_val is None.")
        return
    if X_test is not None and (y_test is None or len(X_test) != len(y_test)):
        logging.warning("      (Warning) Cannot check Overfitting: Test X and y data sizes do not match or y_test is None.")
        X_test, y_test = None, None
    if X_val is not None and len(X_val) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data is empty.")
        return
    if X_train is not None and len(X_train) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Train data is empty.")
        X_train, y_train = None, None
    if X_test is not None and len(X_test) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Test data is empty.")
        X_test, y_test = None, None

    try:
        train_score, val_score, test_score = np.nan, np.nan, np.nan
        model_classes = getattr(model, 'classes_', None)
        if model_classes is None or len(model_classes) < 2:
            logging.warning("      (Warning) Cannot determine model classes or only one class found. Using [0, 1] as fallback.")
            try:
                y_ref = y_val if y_train is None else y_train
                if y_ref is not None:
                    unique_classes = np.unique(y_ref)
                    if len(unique_classes) >= 2:
                        model_classes = unique_classes
                    else:
                        model_classes = [0, 1]
                else:
                    model_classes = [0, 1]
            except Exception:
                model_classes = [0, 1]

        def safe_predict(model, data, method_name):
            if data is None: return None
            if not hasattr(model, method_name):
                logging.warning(f"      (Warning) Model lacks '{method_name}' method.")
                return None
            try:
                pred = getattr(model, method_name)(data)
                if pred is None:
                    logging.warning(f"      (Warning) '{method_name}' returned None.")
                    return None
                if method_name == 'predict_proba':
                    if not isinstance(pred, np.ndarray) or pred.ndim != 2 or pred.shape[1] < 2:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                elif method_name == 'predict':
                    if not isinstance(pred, (np.ndarray, pd.Series)) or pred.ndim != 1:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                return pred
            except Exception as e_pred:
                logging.error(f"      (Error) Calculating {method_name} failed: {e_pred}", exc_info=True)
                return None

        if metric == "AUC":
            val_pred_proba_raw = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba_raw is not None: val_score = roc_auc_score(y_val, val_pred_proba_raw[:, 1])
            train_pred_proba_raw = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba_raw is not None: train_score = roc_auc_score(y_train, train_pred_proba_raw[:, 1])
            test_pred_proba_raw = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba_raw is not None: test_score = roc_auc_score(y_test, test_pred_proba_raw[:, 1])
        elif metric == "LogLoss":
            val_pred_proba = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba is not None: val_score = log_loss(y_val.astype(int), val_pred_proba, labels=model_classes)
            train_pred_proba = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba is not None: train_score = log_loss(y_train.astype(int), train_pred_proba, labels=model_classes)
            test_pred_proba = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba is not None: test_score = log_loss(y_test.astype(int), test_pred_proba, labels=model_classes)
        elif metric == "Accuracy":
            val_pred = safe_predict(model, X_val, "predict")
            if val_pred is not None: val_score = accuracy_score(y_val, val_pred)
            train_pred = safe_predict(model, X_train, "predict")
            if train_pred is not None: train_score = accuracy_score(y_train, train_pred)
            test_pred = safe_predict(model, X_test, "predict")
            if test_pred is not None: test_score = accuracy_score(y_test, test_pred)
        else:
            logging.warning(f"      (Warning) Metric '{metric}' not supported for Overfit check.")
            return

        if pd.notna(train_score): logging.info(f"      Train {metric}: {train_score:.4f}")
        if pd.notna(val_score): logging.info(f"      Val {metric}:   {val_score:.4f}")
        if pd.notna(test_score): logging.info(f"      Test {metric}:  {test_score:.4f}")

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected overfitting logic and logging >>>
        if pd.notna(train_score) and pd.notna(val_score):
            diff_val = train_score - val_score
            diff_val_pct = float('inf') # Default to infinity if denominator is zero
            is_overfitting_val = False
            denominator = 0.0

            if metric == "LogLoss": # Lower is better
                # Check if val_score is significantly worse (higher) than train_score
                denominator = abs(train_score)
                if val_score > train_score + 1e-9: # Check if val score is worse
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True
            else: # Higher is better (AUC, Accuracy)
                # Check if train_score is significantly better (higher) than val_score
                denominator = abs(train_score)
                if train_score > val_score + 1e-9: # Check if train score is better
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True

            # Log comparison results regardless of overfitting detection
            if denominator > 1e-9:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} ({diff_val_pct:.2f}%)")
            else:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_overfitting_val:
                # Use the user-specified log message format
                logging.warning(f"[Patch] Potential Overfitting detected. Train vs Val {metric} gap = {abs(diff_val):.4f} ({diff_val_pct:.2f}% > {threshold_pct:.1f}%)")

        elif X_train is not None:
            logging.warning("      Diff (Train - Val): Cannot calculate (NaN score).")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>

        if pd.notna(val_score) and pd.notna(test_score):
            diff_test = val_score - test_score
            is_generalization_issue = False
            denominator_test = 0.0
            diff_test_pct = float('inf')

            if metric == "LogLoss": # Lower is better
                denominator_test = abs(val_score)
                if test_score > val_score + 1e-9: # Test score is worse
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True
            else: # Higher is better
                denominator_test = abs(val_score)
                if val_score > test_score + 1e-9: # Val score is better
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True

            if denominator_test > 1e-9:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} ({diff_test_pct:.2f}%)")
            else:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_generalization_issue:
                logging.warning(f"      (ALERT) Potential Generalization Issue: Val {metric} significantly {'lower' if metric=='LogLoss' else 'higher'} than Test {metric} (Diff % > {threshold_pct:.1f}%).")
        elif X_test is not None:
            logging.warning("      Diff (Val - Test): Cannot calculate (NaN score).")

    except Exception as e:
        logging.error(f"      (Error) Error during Overfitting check ({metric}): {e}", exc_info=True)

# [Patch v5.0.2] Exclude SHAP noise check from coverage
def check_feature_noise_shap(shap_values, feature_names, threshold=0.01):  # pragma: no cover
    """
    Checks for potentially noisy features based on low mean absolute SHAP values.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info("   [Check] Checking for Feature Noise (SHAP)...")
    if shap_values is None or not isinstance(shap_values, np.ndarray) or not feature_names or not isinstance(feature_names, list) or \
       shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names) or shap_values.shape[0] == 0:
        logging.warning("      (Warning) Skipping Feature Noise Check: Invalid inputs."); return

    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) Found NaN/Inf in Mean Abs SHAP. Skipping noise check.")
            return

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        shap_df["Normalized_SHAP"] = (shap_df["Mean_Abs_SHAP"] / total_shap) if total_shap > 1e-9 else 0.0

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected logging as per user prompt and test failure >>>
        # Use the DataFrame index directly if feature_names match the index
        shap_series_for_check = pd.Series(shap_df["Normalized_SHAP"].values, index=shap_df["Feature"])
        noise_feats = shap_series_for_check[shap_series_for_check < threshold].index.tolist()
        if noise_feats:
            # Use the user-specified log message format (logging.info as per patch 6 note)
            logging.info(f"[Patch] SHAP Noise features detected: {noise_feats}")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>
        else:
            logging.info(f"      (Success) No features with significant noise detected (Normalized SHAP < {threshold:.4f}).")
    except Exception as e:
        logging.error(f"      (Error) Error during Feature Noise check (SHAP): {e}", exc_info=True)

# --- SHAP Analysis Function ---
# [Patch v5.0.2] Exclude SHAP importance analysis from coverage
def analyze_feature_importance_shap(model, model_type, data_sample, features, output_dir, fold_idx=None):  # pragma: no cover
    """
    Analyzes feature importance using SHAP values and saves summary plots.
    (v4.8.8 Patch 1: Enhanced robustness for SHAP value structure and feature validation)
    """
    global shap
    if not shap:
        logging.warning("   (Warning) Skipping SHAP: 'shap' library not found.")
        return
    if model is None:
        logging.warning("   (Warning) Skipping SHAP: Model is None.")
        return
    if data_sample is None or not isinstance(data_sample, pd.DataFrame) or data_sample.empty:
        logging.warning("   (Warning) Skipping SHAP: No sample data.")
        return
    if not features or not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        logging.warning("   (Warning) Skipping SHAP: Invalid features list.")
        return
    if not output_dir or not os.path.isdir(output_dir):
        logging.warning(f"   (Warning) Skipping SHAP: Output directory '{output_dir}' invalid.")
        return

    fold_suffix = f"_fold{fold_idx+1}" if fold_idx is not None else "_validation_set"
    logging.info(f"\n(Analyzing) SHAP analysis ({model_type} - Sample Size: {len(data_sample)}) - {fold_suffix.replace('_',' ').title()}...")

    missing_features = [f for f in features if f not in data_sample.columns]
    if missing_features:
        logging.error(f"   (Error) Skipping SHAP: Missing features in data_sample: {missing_features}")
        return
    try:
        X_shap = data_sample[features].copy()
    except KeyError as e:
        logging.error(f"   (Error) Skipping SHAP: Feature(s) not found: {e}")
        return
    except Exception as e_select:
        logging.error(f"   (Error) Skipping SHAP: Error selecting features: {e_select}", exc_info=True)
        return

    cat_features_indices = []
    cat_feature_names_shap = []
    potential_cat_cols = ['Pattern_Label', 'session', 'Trend_Zone']
    logging.debug("      Processing categorical features for SHAP...")
    for cat_col in potential_cat_cols:
        if cat_col in X_shap.columns:
            try:
                if X_shap[cat_col].isnull().any():
                    X_shap[cat_col].fillna("Missing", inplace=True)
                X_shap[cat_col] = X_shap[cat_col].astype(str)
                if model_type == "CatBoostClassifier":
                    cat_feature_names_shap.append(cat_col)
            except Exception as e_cat_str:
                logging.warning(f"      (Warning) Could not convert '{cat_col}' to string for SHAP: {e_cat_str}.")

    if model_type == "CatBoostClassifier" and cat_feature_names_shap:
        try:
            cat_features_indices = [X_shap.columns.get_loc(col) for col in cat_feature_names_shap]
            logging.debug(f"         Categorical Feature Indices for SHAP Pool: {cat_features_indices}")
        except KeyError as e_loc:
            logging.error(f"      (Error) Could not locate categorical feature index for SHAP: {e_loc}.")
            cat_features_indices = []

    logging.debug("      Handling NaN/Inf in numeric features for SHAP...")
    numeric_cols_shap = X_shap.select_dtypes(include=np.number).columns
    if X_shap[numeric_cols_shap].isin([np.inf, -np.inf]).any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].replace([np.inf, -np.inf], np.nan)
    if X_shap[numeric_cols_shap].isnull().any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].fillna(0)

    if X_shap.isnull().any().any():
        missing_final = X_shap.columns[X_shap.isnull().any()].tolist()
        logging.error(f"      (Error) Skipping SHAP: NaNs still present after fill in columns: {missing_final}")
        return

    try:
        explainer = None
        shap_values = None
        global CatBoostClassifier, Pool

        logging.debug(f"      Initializing SHAP explainer for model type: {model_type}...")
        if model_type == "CatBoostClassifier" and CatBoostClassifier and Pool:
            shap_pool = Pool(X_shap, label=None, cat_features=cat_features_indices)
            explainer = shap.TreeExplainer(model)
            logging.info(f"      Calculating SHAP values (CatBoost)...")
            shap_values = explainer.shap_values(shap_pool)
        else:
            logging.warning(f"      (Warning) SHAP explainer not supported or library missing for model type: {model_type}")
            return

        shap_values_positive_class = None
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            if isinstance(shap_values[1], np.ndarray) and shap_values[1].ndim == 2:
                shap_values_positive_class = shap_values[1]
            else:
                logging.error(f"      (Error) SHAP values list element 1 has unexpected type/shape: {type(shap_values[1])}, {getattr(shap_values[1], 'shape', 'N/A')}")
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_values_positive_class = shap_values
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            if shap_values.shape[0] >= 2 and shap_values.shape[1] == X_shap.shape[0] and shap_values.shape[2] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[1, :, :]
            elif shap_values.shape[2] >= 2 and shap_values.shape[0] == X_shap.shape[0] and shap_values.shape[1] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[:, :, 1]
            elif shap_values.shape[0] == 1:
                shap_values_positive_class = shap_values[0, :, :]
                logging.warning("      SHAP values have only one class output, using index 0.")
            else:
                logging.error(f"      (Error) Unexpected 3D SHAP values shape: {shap_values.shape}. Cannot determine positive class.")
        else:
            logging.error(f"      (Error) Unexpected SHAP values structure (Type: {type(shap_values)}, Shape: {getattr(shap_values, 'shape', 'N/A')}). Cannot plot.")
            return

        if shap_values_positive_class is None:
            logging.error("      (Error) Could not identify SHAP values for positive class.")
            return
        if shap_values_positive_class.shape[1] != len(features):
            logging.error(f"      (Error) SHAP feature dimension mismatch ({shap_values_positive_class.shape[1]} != {len(features)}). Cannot proceed.")
            return
        if shap_values_positive_class.shape[0] != X_shap.shape[0]:
            logging.error(f"      (Error) SHAP sample dimension mismatch ({shap_values_positive_class.shape[0]} != {X_shap.shape[0]}). Cannot proceed.")
            return

        logging.info("      Creating SHAP Summary Plot (bar type)...")
        shap_plot_path = os.path.join(output_dir, f"shap_summary_{model_type}_bar{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="bar", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Importance ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Bar): {os.path.basename(shap_plot_path)}")
        except Exception as e_save_bar:
            logging.error(f"      (Error) Failed to create/save SHAP bar plot: {e_save_bar}", exc_info=True)
        finally:
            plt.close()

        logging.info("      Creating SHAP Summary Plot (beeswarm/dot type)...")
        shap_beeswarm_path = os.path.join(output_dir, f"shap_summary_{model_type}_beeswarm{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="dot", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Summary ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Beeswarm): {os.path.basename(shap_beeswarm_path)}")
        except Exception as e_save_beeswarm:
            logging.error(f"      (Error) Failed to create/save SHAP beeswarm plot: {e_save_beeswarm}", exc_info=True)
        finally:
            plt.close()

    except ImportError:
        logging.error("   (Error) SHAP Error: Could not import shap library components.")
    except Exception as e:
        logging.error(f"   (Error) Error during SHAP analysis: {e}", exc_info=True)

# --- Feature Loading Function ---
# [Patch v5.0.2] Exclude feature loader from coverage
def load_features_for_model(model_name, output_dir):  # pragma: no cover
    """
    Loads the feature list for a specific model purpose from a JSON file.
    Falls back to loading 'features_main.json' if the specific file is not found.
    """
    features_filename = f"features_{model_name}.json"
    features_file_path = os.path.join(output_dir, features_filename)
    logging.info(f"   (Feature Load) Attempting to load features for '{model_name}' from: {features_file_path}")

    if not os.path.exists(features_file_path):
        logging.info(
            f"   (Info) Feature file not found for model '{model_name}': {os.path.basename(features_file_path)}"
        )
        main_features_path = os.path.join(output_dir, "features_main.json")
        if model_name != "main" and os.path.exists(main_features_path):
            logging.info(
                "      (Fallback) Loading features from 'features_main.json' instead."
            )
            features_file_path = main_features_path  # Use main path for fallback
        else:
            logging.info(
                "      (Generating) Default features_main.json."
            )
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(main_features_path, "w", encoding="utf-8") as f_def:
                    json.dump(DEFAULT_META_CLASSIFIER_FEATURES, f_def, ensure_ascii=False, indent=2)
                logging.info(
                    "      (Generated) Default features_main.json created using DEFAULT_META_CLASSIFIER_FEATURES."
                )
                features_file_path = main_features_path
            except Exception as e_write:
                logging.error(
                    f"   (Error) Could not create default features_main.json: {e_write}"
                )
                return DEFAULT_META_CLASSIFIER_FEATURES

    try:
        features = load_json_with_comments(features_file_path)
        if isinstance(features, list) and all(isinstance(feat, str) for feat in features):
            logging.info(f"      (Success) Loaded {len(features)} features for model '{model_name}' from '{os.path.basename(features_file_path)}'.")
            return features
        else:
            logging.error(f"   (Error) Invalid format in feature file: {features_file_path}. Expected list of strings.")
            return None
    except json.JSONDecodeError as e_json:
        logging.error(f"   (Error) Failed to decode JSON from feature file '{os.path.basename(features_file_path)}': {e_json}")
        return None
    except Exception as e:
        logging.error(f"   (Error) Failed to load features for model '{model_name}' from '{os.path.basename(features_file_path)}': {e}", exc_info=True)
        return None

# --- Model Switcher Function ---
# [Patch v5.0.2] Exclude model switcher from coverage
def select_model_for_trade(context, available_models=None):  # pragma: no cover
    """
    Selects the appropriate AI model ('main', 'spike', 'cluster') based on context.
    Falls back to 'main' if the selected model is invalid or missing.
    """
    selected_model_key = 'main' # Default model
    confidence = None

    cluster_value = context.get('cluster')
    spike_score_value = context.get('spike_score', 0.0)

    if not isinstance(cluster_value, (int, float, np.number)) or pd.isna(cluster_value):
        cluster_value = None
    if not isinstance(spike_score_value, (int, float, np.number)) or pd.isna(spike_score_value):
        spike_score_value = 0.0

    spike_switch_threshold = 0.6
    cluster_switch_value = 2

    logging.debug(f"      (Switcher) Context: SpikeScore={spike_score_value:.3f}, Cluster={cluster_value}")

    if spike_score_value > spike_switch_threshold:
        selected_model_key = 'spike'
        confidence = spike_score_value
    elif cluster_value == cluster_switch_value:
        selected_model_key = 'cluster'
        confidence = 0.8
    else:
        selected_model_key = 'main'
        confidence = None

    if available_models is None:
        logging.error("      (Switcher Error) 'available_models' is None. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None
    elif selected_model_key not in available_models or \
         available_models.get(selected_model_key, {}).get('model') is None or \
         not available_models.get(selected_model_key, {}).get('features'):
        logging.warning(f"      (Switcher Warning) Selected model '{selected_model_key}' invalid/missing. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None

    logging.debug(f"      (Switcher) Final Selected Model: '{selected_model_key}', Confidence: {confidence}")
    return selected_model_key, confidence

logging.info("Part 6: Machine Learning Configuration & Helpers Loaded (v4.8.8 Patch 9 Applied).")
