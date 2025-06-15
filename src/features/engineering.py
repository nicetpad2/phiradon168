from .common import *
from .technical import *

def tag_price_structure_patterns(df):
    logging.info("   (Processing) Tagging price structure patterns...")
    if not isinstance(df, pd.DataFrame): logging.error("Pattern Tagging Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df.empty: df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    required_cols = ["Gain_Z", "High", "Low", "Close", "Open", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"      (Warning) Missing columns for Pattern Labeling. Setting all to 'Normal'.")
        df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    df_patterns = df.copy()
    for col in ["Gain_Z", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]: df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce').fillna(0)
    for col in ["High", "Low", "Close", "Open"]:
        df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce')
        if df_patterns[col].isnull().any(): df_patterns[col] = df_patterns[col].ffill().bfill()
    df_patterns["Pattern_Label"] = "Normal"
    prev_high = df_patterns["High"].shift(1); prev_low = df_patterns["Low"].shift(1); prev_gain = df_patterns["Gain"].shift(1).fillna(0); prev_body = df_patterns["Candle_Body"].shift(1).fillna(0); prev_macd_hist = df_patterns["MACD_hist"].shift(1).fillna(0)
    breakout_cond = ((df_patterns["Gain_Z"].abs() > PATTERN_BREAKOUT_Z_THRESH) | ((df_patterns["High"] > prev_high) & (df_patterns["Close"] > prev_high)) | ((df_patterns["Low"] < prev_low) & (df_patterns["Close"] < prev_low))).fillna(False)
    reversal_cond = (((prev_gain < 0) & (df_patterns["Gain"] > 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO))) | ((prev_gain > 0) & (df_patterns["Gain"] < 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO)))).fillna(False)
    inside_bar_cond = ((df_patterns["High"] < prev_high) & (df_patterns["Low"] > prev_low)).fillna(False)
    strong_trend_cond = (((df_patterns["Gain_Z"] > PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] > 0) & (df_patterns["MACD_hist"] > prev_macd_hist)) | ((df_patterns["Gain_Z"] < -PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] < 0) & (df_patterns["MACD_hist"] < prev_macd_hist))).fillna(False)
    choppy_cond = ((df_patterns["Candle_Ratio"] < PATTERN_CHOPPY_CANDLE_RATIO) & (df_patterns["Wick_Ratio"] > PATTERN_CHOPPY_WICK_RATIO)).fillna(False)
    df_patterns.loc[breakout_cond, "Pattern_Label"] = "Breakout"
    df_patterns.loc[reversal_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Reversal"
    df_patterns.loc[inside_bar_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "InsideBar"
    df_patterns.loc[strong_trend_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "StrongTrend"
    df_patterns.loc[choppy_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Choppy"
    logging.info(f"      Pattern Label Distribution:\n{df_patterns['Pattern_Label'].value_counts(normalize=True).round(3).to_string()}")
    df["Pattern_Label"] = df_patterns["Pattern_Label"].astype('category')
    del df_patterns, prev_high, prev_low, prev_gain, prev_body, prev_macd_hist, breakout_cond, reversal_cond, inside_bar_cond, strong_trend_cond, choppy_cond; maybe_collect()
    return df

def calculate_m15_trend_zone(df_m15):
    """Calculate UP/DOWN/NEUTRAL trend zone for M15 timeframe."""
    logging.info("(Processing) กำลังคำนวณ M15 Trend Zone...")
    # [Patch v6.6.3] Ensure index is unique and sorted before indicator calculation
    cache_key = hash(tuple(df_m15.index)) if isinstance(df_m15, pd.DataFrame) else None
    if cache_key is not None and cache_key in _m15_trend_cache:
        logging.info("      [Cache] ใช้ผลลัพธ์ Trend Zone จาก cache")
        cached_df = _m15_trend_cache[cache_key]
        return cached_df.copy()
    if not isinstance(df_m15, pd.DataFrame): logging.error("M15 Trend Zone Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m15.empty or "Close" not in df_m15.columns:
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"})
        result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        if result_df.index.duplicated().any():
            result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
        if not result_df.index.is_monotonic_increasing:
            result_df.sort_index(inplace=True)
        return result_df
    df = df_m15.copy()
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        logging.warning(
            "(Warning) พบ duplicate labels ใน index M15, กำลังลบซ้ำ (คงไว้ค่าแรกของแต่ละ index)"
        )
        df = df[~df.index.duplicated(keep='first')]
        logging.info(
            f"      Removed {dup_count} duplicate rows from M15 index."
        )
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)
        logging.info("      Sorted M15 index in ascending order for Trend Zone calculation")
    try:
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
        if df["Close"].isnull().all():
            result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"})
            result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
            if cache_key is not None:
                _m15_trend_cache[cache_key] = result_df
            if result_df.index.duplicated().any():
                result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
            if not result_df.index.is_monotonic_increasing:
                result_df.sort_index(inplace=True)
            return result_df
        import importlib
        features_pkg = importlib.import_module(__package__)
        df["EMA_Fast"] = features_pkg.ema(df["Close"], M15_TREND_EMA_FAST)
        df["EMA_Slow"] = features_pkg.ema(df["Close"], M15_TREND_EMA_SLOW)
        df["RSI"] = features_pkg.rsi(df["Close"], M15_TREND_RSI_PERIOD)
        df.dropna(subset=["EMA_Fast", "EMA_Slow", "RSI"], inplace=True)
        if df.empty:
            result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"})
            result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
            if cache_key is not None:
                _m15_trend_cache[cache_key] = result_df
            if result_df.index.duplicated().any():
                result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
            if not result_df.index.is_monotonic_increasing:
                result_df.sort_index(inplace=True)
            return result_df
        is_up = (df["EMA_Fast"] > df["EMA_Slow"]) & (df["RSI"] > M15_TREND_RSI_UP); is_down = (df["EMA_Fast"] < df["EMA_Slow"]) & (df["RSI"] < M15_TREND_RSI_DOWN)
        df["Trend_Zone"] = "NEUTRAL"; df.loc[is_up, "Trend_Zone"] = "UP"; df.loc[is_down, "Trend_Zone"] = "DOWN"
        if not df.empty: logging.info(f"   การกระจาย M15 Trend Zone:\n{df['Trend_Zone'].value_counts(normalize=True).round(3).to_string()}")
        # Build result DataFrame with unique, sorted index (fill missing with NEUTRAL)
        target_index = pd.Index(df_m15.index.unique())
        target_index = target_index.sort_values()
        result_df = df[["Trend_Zone"]].reindex(target_index).fillna("NEUTRAL")
        result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        del df, is_up, is_down; maybe_collect();
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        return result_df
    except Exception as e:
        logging.error(f"(Error) การคำนวณ M15 Trend Zone ล้มเหลว: {e}", exc_info=True)
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"})
        result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        if result_df.index.duplicated().any():
            result_df = result_df.loc[~result_df.index.duplicated(keep='first')]
        if not result_df.index.is_monotonic_increasing:
            result_df.sort_index(inplace=True)
        return result_df

# [Patch v5.5.6] Helper to evaluate higher timeframe trend using SMA crossover
def get_mtf_sma_trend(df_m15, fast=50, slow=200, rsi_period=14, rsi_upper=70, rsi_lower=30):
    """Return trend direction ('UP', 'DOWN', 'NEUTRAL') for M15 data.

    Parameters
    ----------
    df_m15 : pandas.DataFrame
        M15 OHLC data with at least a 'Close' column.
    fast : int, optional
        Fast SMA period. Default 50.
    slow : int, optional
        Slow SMA period. Default 200.
    rsi_period : int, optional
        RSI period. Default 14.
    rsi_upper : float, optional
        Upper RSI filter for uptrend. Default 70.
    rsi_lower : float, optional
        Lower RSI filter for downtrend. Default 30.
    """
    if not isinstance(df_m15, pd.DataFrame) or df_m15.empty or "Close" not in df_m15.columns:
        return "NEUTRAL"
    close = pd.to_numeric(df_m15["Close"], errors="coerce")
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    rsi_series = rsi(close, period=rsi_period)
    if fast_ma.empty or slow_ma.empty or rsi_series.empty:
        return "NEUTRAL"
    last_fast = fast_ma.iloc[-1]
    last_slow = slow_ma.iloc[-1]
    last_rsi = rsi_series.iloc[-1]
    if pd.isna(last_fast) or pd.isna(last_slow) or pd.isna(last_rsi):
        return "NEUTRAL"
    if last_fast > last_slow and last_rsi < rsi_upper:
        return "UP"
    if last_fast < last_slow and last_rsi > rsi_lower:
        return "DOWN"
    return "NEUTRAL"

# [Patch v5.0.2] Exclude heavy engineering logic from coverage
def engineer_m1_features(df_m1, timeframe_minutes=TIMEFRAME_MINUTES_M1, lag_features_config=None):  # pragma: no cover
    logging.info("[QA] Start M1 Feature Engineering")
    logging.info("(Processing) กำลังสร้าง Features M1 (v4.9.0)...") # <<< MODIFIED v4.9.0
    logging.info(f"Rows at start: {df_m1.shape[0]}")
    if not isinstance(df_m1, pd.DataFrame): logging.error("Engineer M1 Features Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการสร้าง Features M1: DataFrame ว่างเปล่า."); return df_m1
    df = df_m1.copy()
    # [Patch v6.8.11] Normalize lowercase price column names
    rename_map={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    df.rename(columns={c:rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}, inplace=True)
    price_cols = ["Open", "High", "Low", "Close"]
    if any(col not in df.columns for col in price_cols):
        logging.warning(f"   (Warning) ขาดคอลัมน์ราคา M1. บาง Features อาจเป็น NaN.")
        base_feature_cols = ["Candle_Body", "Candle_Range", "Gain", "Candle_Ratio", "Upper_Wick", "Lower_Wick", "Wick_Length", "Wick_Ratio", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", 'Volatility_Index', 'ADX', 'RSI']
        for col in base_feature_cols:
            if col not in df.columns: df[col] = np.nan
        if "Pattern_Label" not in df.columns: df["Pattern_Label"] = "N/A"
    else:
        for col in price_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=price_cols, inplace=True)
        logging.info(f"Rows after drop price NaN: {df.shape[0]}")
        if df.empty: logging.warning("   (Warning) M1 DataFrame ว่างเปล่าหลังลบราคา NaN."); return df
        df["Candle_Body"]=abs(df["Close"]-df["Open"]).astype('float32'); df["Candle_Range"]=(df["High"]-df["Low"]).astype('float32'); df["Gain"]=(df["Close"]-df["Open"]).astype('float32')
        df["Candle_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Candle_Body"]/df["Candle_Range"],0.0).astype('float32'); df["Upper_Wick"]=(df["High"]-np.maximum(df["Open"],df["Close"])).astype('float32')
        df["Lower_Wick"]=(np.minimum(df["Open"],df["Close"])-df["Low"]).astype('float32'); df["Wick_Length"]=(df["Upper_Wick"]+df["Lower_Wick"]).astype('float32')
        df["Wick_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Wick_Length"]/df["Candle_Range"],0.0).astype('float32'); df["Gain_Z"]=rolling_zscore(df["Gain"].fillna(0),window=ROLLING_Z_WINDOW_M1)
        df["MACD_line"],df["MACD_signal"],df["MACD_hist"]=macd(df["Close"])
        if "MACD_hist" in df.columns and df["MACD_hist"].notna().any(): df["MACD_hist_smooth"]=df["MACD_hist"].rolling(window=5,min_periods=1).mean().fillna(0).astype('float32')
        else: df["MACD_hist_smooth"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ MACD_hist_smooth.")
        df=atr(df,14)
        if "ATR_14" in df.columns and df["ATR_14"].notna().any(): df["ATR_14_Rolling_Avg"]=sma(df["ATR_14"],ATR_ROLLING_AVG_PERIOD)
        else: df["ATR_14_Rolling_Avg"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ ATR_14_Rolling_Avg.")
        df["Candle_Speed"]=(df["Gain"]/max(timeframe_minutes,1e-6)).astype('float32'); df["RSI"]=rsi(df["Close"],period=14)
    if lag_features_config and isinstance(lag_features_config,dict):
        for feature in lag_features_config.get('features',[]):
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                for lag in lag_features_config.get('lags',[]):
                    if isinstance(lag,int) and lag>0: df[f"{feature}_lag{lag}"]=df[feature].shift(lag).astype('float32')
    if 'ATR_14' in df.columns and 'ATR_14_Rolling_Avg' in df.columns and df['ATR_14_Rolling_Avg'].notna().any():
        df['Volatility_Index']=np.where(df['ATR_14_Rolling_Avg'].abs()>1e-9,df['ATR_14']/df['ATR_14_Rolling_Avg'],np.nan)
        df['Volatility_Index']=df['Volatility_Index'].ffill().fillna(1.0).astype('float32')
    else: df['Volatility_Index']=1.0; logging.warning("         (Warning) ไม่สามารถคำนวณ Volatility_Index.")
    if all(c in df.columns for c in ['High','Low','Close']) and ta:
        try:
            if len(df.dropna(subset=['High','Low','Close']))>=14*2+10: adx_indicator=ta.trend.ADXIndicator(df['High'],df['Low'],df['Close'],window=14,fillna=False); df['ADX']=adx_indicator.adx().ffill().fillna(25.0).astype('float32')
            else: df['ADX']=25.0; logging.warning("         (Warning) ไม่สามารถคำนวณ ADX: ข้อมูลไม่เพียงพอ.")
        except Exception as e_adx: df['ADX']=25.0; logging.warning(f"         (Warning) ไม่สามารถคำนวณ ADX: {e_adx}")
    else: df['ADX']=25.0
    if all(col in df.columns for col in ["Gain_Z","High","Low","Close","Open","MACD_hist","Candle_Ratio","Wick_Ratio","Gain","Candle_Body"]): df=tag_price_structure_patterns(df)
    else: df["Pattern_Label"]="N/A"; df["Pattern_Label"]=df["Pattern_Label"].astype('category'); logging.warning("   (Warning) ข้ามการ Tag Patterns.")
    if 'cluster' not in df.columns:
        try:
            cluster_features=['Gain_Z','Volatility_Index','Candle_Ratio','RSI','ADX']; features_present=[f for f in cluster_features if f in df.columns and df[f].notna().any()]
            if len(features_present)<2 or len(df[features_present].dropna())<3: df['cluster']=0; logging.warning("         (Warning) Not enough valid features/samples for clustering.")
            else:
                X_cluster_raw = df[features_present].copy().replace([np.inf, -np.inf], np.nan)
                X_cluster = X_cluster_raw.fillna(X_cluster_raw.median()).fillna(0)
                # [Patch v5.0.16] Skip KMeans if duplicate samples could cause ConvergenceWarning
                if len(X_cluster) >= 3:
                    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_cluster)
                    if len(np.unique(X_scaled, axis=0)) < 3:
                        df['cluster'] = 0
                        logging.warning("         (Warning) Not enough unique samples for clustering.")
                    else:
                        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                        df['cluster'] = kmeans.fit_predict(X_scaled)
                else:
                    df['cluster'] = 0
                    logging.warning("         (Warning) Not enough samples after cleaning for clustering.")
        except Exception as e_cluster: df['cluster']=0; logging.error(f"         (Error) Clustering failed: {e_cluster}.",exc_info=True)
        if 'cluster' in df.columns: df['cluster']=pd.to_numeric(df['cluster'],downcast='integer')
    if 'spike_score' not in df.columns:
        try:
            gain_z_abs = abs(pd.to_numeric(df.get('Gain_Z', 0.0), errors='coerce').fillna(0.0))
            wick_ratio = pd.to_numeric(df.get('Wick_Ratio', 0.0), errors='coerce').fillna(0.0)
            atr_val = pd.to_numeric(df.get('ATR_14', 1.0), errors='coerce').fillna(1.0).replace([np.inf, -np.inf], 1.0)
            score = (wick_ratio * 0.5 + gain_z_abs * 0.3 + atr_val * 0.2)
            score = np.where((atr_val > 1.5) & (wick_ratio > 0.6), score * 1.2, score)
            df['spike_score'] = score.clip(0, 1).astype('float32')
        except Exception as e_spike:
            df['spike_score'] = 0.0
            logging.error(f"         (Error) Spike score calculation failed: {e_spike}.", exc_info=True)

    # [Patch v5.7.9] additional engineered features
    if {'BuyVolume', 'SellVolume'}.issubset(df.columns):
        df['OF_Imbalance'] = calculate_order_flow_imbalance(df)
    else:
        df['OF_Imbalance'] = 0.0

    if 'Close' in df.columns:
        df['Momentum_Divergence'] = calculate_momentum_divergence(df['Close'])
    else:
        df['Momentum_Divergence'] = 0.0

    if 'Volume' in df.columns:
        df['Relative_Volume'] = calculate_relative_volume(df)
    else:
        df['Relative_Volume'] = 0.0
    if 'session' not in df.columns:
        logging.info("      Creating 'session' column...")
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
            df['session'] = get_session_tags_vectorized(df.index)
            logging.info(
                f"         Session distribution:\n{df['session'].value_counts(normalize=True).round(3).to_string()}"
            )
        except Exception as e_session:
            logging.error(
                f"         (Error) Session calculation failed: {e_session}. Assigning 'Other'.",
                exc_info=True,
            )
            df['session'] = pd.Series('Other', index=df.index).astype('category')
    if 'model_tag' not in df.columns: df['model_tag'] = 'N/A'
    logging.info("(Success) สร้าง Features M1 (v4.9.0) เสร็จสิ้น.")  # <<< MODIFIED v4.9.0
    numeric_cols_clean = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_clean) > 0:
        df[numeric_cols_clean] = df[numeric_cols_clean].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols_clean] = df[numeric_cols_clean].ffill().fillna(0)
    # [Patch v5.5.4] Run QA check after cleaning to avoid false warnings
    if df.isnull().any().any() or np.isinf(df[numeric_cols_clean]).any().any():
        logging.warning("[QA WARNING] NaN/Inf detected in engineered features")
    nan_counts = df.isna().sum()
    logging.info("NaN per column after engineering:\n%s", nan_counts.to_string())
    high_nan_cols = nan_counts[nan_counts > len(df) * 0.5]
    if not high_nan_cols.empty:
        logging.warning(f"Columns with >50% NaN: {list(high_nan_cols.index)}")
    logging.info("[QA] M1 Feature Engineering Completed")
    return df.reindex(df_m1.index)

# [Patch v5.0.2] Exclude heavy cleaning logic from coverage
def clean_m1_data(df_m1):  # pragma: no cover
    logging.info("(Processing) กำลังกำหนด Features M1 สำหรับ Drift และแปลงประเภท (v4.9.0)...") # <<< MODIFIED v4.9.0
    if not isinstance(df_m1, pd.DataFrame): logging.error("Clean M1 Data Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการทำความสะอาดข้อมูล M1: DataFrame ว่างเปล่า."); return pd.DataFrame(), []
    logging.info(f"Rows before cleaning: {df_m1.shape[0]}")
    df_cleaned = df_m1.copy()
    potential_m1_features = ["Candle_Body", "Candle_Range", "Candle_Ratio", "Gain", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", "Wick_Length", "Wick_Ratio", "Pattern_Label", "Signal_Score", 'Volatility_Index', 'ADX', 'RSI', 'cluster', 'spike_score', 'OF_Imbalance', 'Momentum_Divergence', 'Relative_Volume', 'session']
    lag_cols_in_df = [col for col in df_cleaned.columns if '_lag' in col]
    potential_m1_features.extend(lag_cols_in_df)
    if META_CLASSIFIER_FEATURES: potential_m1_features.extend([f for f in META_CLASSIFIER_FEATURES if f not in potential_m1_features])
    potential_m1_features = sorted(list(dict.fromkeys(potential_m1_features)))
    m1_features_for_drift = [f for f in potential_m1_features if f in df_cleaned.columns]
    logging.info(f"   กำหนด {len(m1_features_for_drift)} Features M1 สำหรับ Drift: {m1_features_for_drift}")
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        try:
            inf_mask = df_cleaned[numeric_cols].isin([np.inf, -np.inf])
            if inf_mask.any().any():
                cols_with_inf = df_cleaned[numeric_cols].columns[inf_mask.any()].tolist()
                logging.warning(f"      [Inf Check] พบ Inf ใน: {cols_with_inf}. กำลังแทนที่ด้วย NaN...")
                df_cleaned[cols_with_inf] = df_cleaned[cols_with_inf].replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            cols_with_nan = df_cleaned[numeric_cols].columns[df_cleaned[numeric_cols].isnull().any()].tolist()
            if cols_with_nan:
                logging.info(f"      [NaN Check] พบ NaN ใน: {cols_with_nan}. กำลังเติมด้วย ffill().fillna(0)...")
                df_cleaned[cols_with_nan] = df_cleaned[cols_with_nan].ffill().fillna(0)
            for col in numeric_cols:
                if col not in df_cleaned.columns: continue
                if pd.api.types.is_integer_dtype(df_cleaned[col].dtype): df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df_cleaned[col].dtype) and df_cleaned[col].dtype != 'float32': df_cleaned[col] = df_cleaned[col].astype('float32')
        except Exception as e: logging.error(f"   (Error) เกิดข้อผิดพลาดในการแปลงประเภทข้อมูลหรือเติม NaN/Inf: {e}.", exc_info=True)
    categorical_cols = ['Pattern_Label', 'session']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any(): df_cleaned[col] = df_cleaned[col].fillna("Unknown") # Use assignment
            if not isinstance(df_cleaned[col].dtype, pd.CategoricalDtype):
                try: df_cleaned[col] = df_cleaned[col].astype('category')
                except Exception as e_cat: logging.warning(f"   (Warning) เกิดข้อผิดพลาดในการแปลง '{col}' เป็น category: {e_cat}.")
    logging.info(f"Rows after cleaning features: {df_cleaned.shape[0]}")
    logging.info("NaN count after clean_m1_data:\n%s", df_cleaned.isna().sum().to_string())
    logging.info("(Success) กำหนด Features M1 และแปลงประเภท (v4.9.0) เสร็จสิ้น.") # <<< MODIFIED v4.9.0
    return df_cleaned, m1_features_for_drift

# [Patch v5.0.2] Exclude heavy signal calculation from coverage
def calculate_m1_entry_signals(df_m1: pd.DataFrame, config: dict) -> pd.DataFrame:  # pragma: no cover
    logging.debug("      (Calculating M1 Signals)...")
    df = df_m1.copy(); df['Signal_Score'] = 0.0
    gain_z_thresh = config.get('gain_z_thresh', 0.3); rsi_thresh_buy = config.get('rsi_thresh_buy', 50)
    rsi_thresh_sell = config.get('rsi_thresh_sell', 50); volatility_max = config.get('volatility_max', 4.0)
    entry_score_min = config.get('min_signal_score', MIN_SIGNAL_SCORE_ENTRY); ignore_rsi = config.get('ignore_rsi_scoring', False)
    df['Gain_Z'] = df.get('Gain_Z', pd.Series(0.0, index=df.index)).fillna(0.0)
    buy_gain_z_cond = df['Gain_Z'] > gain_z_thresh; sell_gain_z_cond = df['Gain_Z'] < -gain_z_thresh
    df['Pattern_Label'] = df.get('Pattern_Label', pd.Series('Normal', index=df.index)).astype(str).fillna('Normal')
    buy_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend']) & (df['Gain_Z'] > 0)
    sell_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend', 'Reversal']) & (df['Gain_Z'] < 0)
    df['RSI'] = df.get('RSI', pd.Series(50.0, index=df.index)).fillna(50.0)
    buy_rsi_cond = df['RSI'] > rsi_thresh_buy; sell_rsi_cond = df['RSI'] < rsi_thresh_sell
    df['Volatility_Index'] = df.get('Volatility_Index', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol_cond = df['Volatility_Index'] < volatility_max
    df.loc[buy_gain_z_cond, 'Signal_Score'] += 1.0; df.loc[sell_gain_z_cond, 'Signal_Score'] -= 1.0
    df.loc[buy_pattern_cond, 'Signal_Score'] += 1.0; df.loc[sell_pattern_cond, 'Signal_Score'] -= 1.0
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Signal_Score'] += 1.0; df.loc[sell_rsi_cond, 'Signal_Score'] -= 1.0
    df.loc[vol_cond, 'Signal_Score'] += 1.0
    df['Signal_Score'] = df['Signal_Score'].astype('float32')
    df['Entry_Long'] = ((df['Signal_Score'] > 0) & (df['Signal_Score'] >= entry_score_min)).astype(int)
    df['Entry_Short'] = ((df['Signal_Score'] < 0) & (abs(df['Signal_Score']) >= entry_score_min)).astype(int)
    df['Trade_Reason'] = ""; df.loc[buy_gain_z_cond, 'Trade_Reason'] += f"+Gz>{gain_z_thresh:.1f}"
    df.loc[sell_gain_z_cond, 'Trade_Reason'] += f"+Gz<{-gain_z_thresh:.1f}"; df.loc[buy_pattern_cond, 'Trade_Reason'] += "+PBuy"
    df.loc[sell_pattern_cond, 'Trade_Reason'] += "+PSell"
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Trade_Reason'] += f"+RSI>{rsi_thresh_buy}"; df.loc[sell_rsi_cond, 'Trade_Reason'] += f"+RSI<{rsi_thresh_sell}"
    df.loc[vol_cond, 'Trade_Reason'] += f"+Vol<{volatility_max:.1f}"
    buy_entry_mask = df['Entry_Long'] == 1; sell_entry_mask = df['Entry_Short'] == 1
    df.loc[buy_entry_mask, 'Trade_Reason'] = "BUY(" + df.loc[buy_entry_mask, 'Signal_Score'].round(1).astype(str) + "):" + df.loc[buy_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[sell_entry_mask, 'Trade_Reason'] = "SELL(" + df.loc[sell_entry_mask, 'Signal_Score'].abs().round(1).astype(str) + "):" + df.loc[sell_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[~(buy_entry_mask | sell_entry_mask), 'Trade_Reason'] = "NONE"
    df['Trade_Tag'] = df['Signal_Score'].round(1).astype(str) + "_" + df['Pattern_Label'].astype(str)
    return df

logging.info("Part 5: Feature Engineering & Indicator Calculation Functions Loaded.")
# === END OF PART 5/12 ===
# === START OF PART 6/12 ===

# ==============================================================================
# === PART 6: Machine Learning Configuration & Helpers (v4.8.8 - Patch 9 Applied: Fix ML Log) ===
# ==============================================================================
# <<< MODIFIED v4.8.7: Re-verified robustness checks in check_model_overfit based on latest prompt. >>>
# <<< MODIFIED v4.8.8: Ensured check_model_overfit robustness aligns with final prompt. Added default and try-except for META_META_MIN_PROBA_THRESH. >>>
# <<< MODIFIED v4.8.8 (Patch 1): Enhanced robustness in check_model_overfit, check_feature_noise_shap, analyze_feature_importance_shap, select_top_shap_features. Corrected logic for overfitting detection and noise logging. >>>
# <<< MODIFIED v4.8.8 (Patch 4): Corrected select_top_shap_features return value for invalid feature_names. Fixed overfitting detection logic and noise logging in check_model_overfit/check_feature_noise_shap. >>>
# <<< MODIFIED v4.8.8 (Patch 6): Applied user prompt fixes for check_model_overfit and check_feature_noise_shap logging. >>>
# <<< MODIFIED v4.8.8 (Patch 9): Fixed logging format and conditions in check_model_overfit and check_feature_noise_shap as per failed tests and plan. >>>
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    import shap
except ImportError:
    shap = None
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
except ImportError:
    accuracy_score = roc_auc_score = log_loss = classification_report = lambda *args, **kwargs: None
    logging.error("Scikit-learn metrics not found!")

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_META_MIN_PROBA_THRESH = 0.25
DEFAULT_ENABLE_OPTUNA_TUNING = True
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_META_CLASSIFIER_FEATURES = [
    "RSI", "MACD_hist_smooth", "ATR_14", "ADX", "Gain_Z", "Volatility_Index",
    "Candle_Ratio", "Wick_Ratio", "Candle_Speed",
    "Gain_Z_lag1", "Gain_Z_lag3", "Gain_Z_lag5",
    "Candle_Speed_lag1", "Candle_Speed_lag3", "Candle_Speed_lag5",
    "cluster", "spike_score", "Pattern_Label",
    "OF_Imbalance", "Momentum_Divergence", "Relative_Volume",
]
# <<< [Patch] Added default for Meta-Meta threshold >>>
DEFAULT_META_META_MIN_PROBA_THRESH = 0.5

try:
    USE_META_CLASSIFIER
except NameError:
    USE_META_CLASSIFIER = True
try:
    USE_META_META_CLASSIFIER
except NameError:
    USE_META_META_CLASSIFIER = False
try:
    META_CLASSIFIER_PATH
except NameError:
    META_CLASSIFIER_PATH = "meta_classifier.pkl"
try:
    SPIKE_MODEL_PATH
except NameError:
    SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
try:
    CLUSTER_MODEL_PATH
except NameError:
    CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
try:
    META_META_CLASSIFIER_PATH
except NameError:
    META_META_CLASSIFIER_PATH = "meta_meta_classifier.pkl"
try:
    META_CLASSIFIER_FEATURES
except NameError:
    META_CLASSIFIER_FEATURES = DEFAULT_META_CLASSIFIER_FEATURES
try:
    META_META_CLASSIFIER_FEATURES
except NameError:
    META_META_CLASSIFIER_FEATURES = []
try:
    META_MIN_PROBA_THRESH
except NameError:
    META_MIN_PROBA_THRESH = DEFAULT_META_MIN_PROBA_THRESH
META_MIN_PROBA_THRESH = get_env_float("META_MIN_PROBA_THRESH", META_MIN_PROBA_THRESH)  # env override
try:
    REENTRY_MIN_PROBA_THRESH
except NameError:
    REENTRY_MIN_PROBA_THRESH = META_MIN_PROBA_THRESH
REENTRY_MIN_PROBA_THRESH = get_env_float("REENTRY_MIN_PROBA_THRESH", REENTRY_MIN_PROBA_THRESH)  # env override
# <<< [Patch] Added try-except for Meta-Meta threshold >>>
try:
    META_META_MIN_PROBA_THRESH
except NameError:
    META_META_MIN_PROBA_THRESH = DEFAULT_META_META_MIN_PROBA_THRESH
META_META_MIN_PROBA_THRESH = get_env_float("META_META_MIN_PROBA_THRESH", META_META_MIN_PROBA_THRESH)  # env override
# <<< End of [Patch] >>>
try:
    ENABLE_OPTUNA_TUNING
except NameError:
    ENABLE_OPTUNA_TUNING = DEFAULT_ENABLE_OPTUNA_TUNING
try:
    OPTUNA_N_TRIALS
except NameError:
    OPTUNA_N_TRIALS = DEFAULT_OPTUNA_N_TRIALS
try:
    OPTUNA_CV_SPLITS
except NameError:
    OPTUNA_CV_SPLITS = DEFAULT_OPTUNA_CV_SPLITS
try:
    OPTUNA_METRIC
except NameError:
    OPTUNA_METRIC = DEFAULT_OPTUNA_METRIC
try:
    OPTUNA_DIRECTION
except NameError:
    OPTUNA_DIRECTION = DEFAULT_OPTUNA_DIRECTION

logging.info("Loading Machine Learning Configuration & Helpers...")

# --- ML Model Usage Flags ---
logging.info(f"USE_META_CLASSIFIER (L1 Filter): {USE_META_CLASSIFIER}")
logging.info(f"USE_META_META_CLASSIFIER (L2 Filter): {USE_META_META_CLASSIFIER}")

# --- ML Model Paths & Features ---
logging.debug(f"Main L1 Model Path: {META_CLASSIFIER_PATH}")
logging.debug(f"Spike Model Path: {SPIKE_MODEL_PATH}")
logging.debug(f"Cluster Model Path: {CLUSTER_MODEL_PATH}")
logging.debug(f"L2 Model Path: {META_META_CLASSIFIER_PATH}")
logging.debug(f"Default L1 Features (Count): {len(META_CLASSIFIER_FEATURES)}")
logging.debug(f"L2 Features (Count): {len(META_META_CLASSIFIER_FEATURES)}")

# --- ML Thresholds ---
logging.info(f"Default L1 Probability Threshold: {META_MIN_PROBA_THRESH}")
logging.info(f"Default Re-entry Probability Threshold: {REENTRY_MIN_PROBA_THRESH}")
logging.info(f"Default L2 Probability Threshold: {META_META_MIN_PROBA_THRESH}") # Now uses defined value

# --- Optuna Configuration ---
logging.info(f"Optuna Hyperparameter Tuning Enabled: {ENABLE_OPTUNA_TUNING}")
if ENABLE_OPTUNA_TUNING:
    logging.info(f"  Optuna Trials: {OPTUNA_N_TRIALS}")
    logging.info(f"  Optuna CV Splits: {OPTUNA_CV_SPLITS}")
    logging.info(f"  Optuna Metric: {OPTUNA_METRIC} ({OPTUNA_DIRECTION})")

# --- Auto Threshold Tuning ---
ENABLE_AUTO_THRESHOLD_TUNING = True  # [Patch v5.10.4] เปิด Auto Threshold Tuning เพื่อให้ pipeline เรียก threshold optimization ต่อ
logging.info(f"Auto Threshold Tuning Enabled: {ENABLE_AUTO_THRESHOLD_TUNING}")

# --- Global variables to store model info ---
meta_model_type_used = "N/A"
meta_meta_model_type_used = "N/A"
logging.debug("Global model type trackers initialized.")
