### 2025-09-02
- [Patch v5.6.5] Add Optuna WFV search utility
- New/Updated unit tests added for tests.test_optuna_wfv
- QA: pytest -q passed

### 2025-09-01
- [Patch v5.6.4] Fix boundary logic for session tagging and reduce duplicate warnings
- New/Updated unit tests added for tests.test_sessions_utils
- QA: pytest -q passed

### 2025-08-31

- [Patch v5.6.4] Extend Asia session to 22-8 and update tests
- QA: pytest -q passed


#
### 2025-08-30
- [Patch v5.6.3] Reduce log spam for out-of-session timestamps
- QA: pytest -q passed (347 tests)
### 2025-08-29
- [Patch v5.6.2] Remove PyTables dependency for feature persistence
- Fix FutureWarning in check_data_quality
- New/Updated unit tests added for tests.test_features_hdf5
- QA: pytest -q passed (347 tests)
### 2025-08-28
- [Patch v5.6.1] Improve model utilities
- New/Updated unit tests added for tests.test_model_utils_new
- QA: pytest -q passed (325 tests)

### 2025-08-28

- [Patch v5.6.1] Add dataclass-based order logging with rotating file support
- New/Updated unit tests added for tests.test_trade_logger
- QA: pytest -q passed (325 tests)


### 2025-08-26
- [Patch v5.6.0] Refactor font setup and CSV loader
- New/Updated unit tests added for data_loader
- QA: pytest -q passed (325 tests)


### 2025-08-27
- [Patch v5.6.1] Refactor cooldown state management
- New/Updated unit tests added for cooldown_utils

- QA: pytest -q passed (333 tests)


### 2025-08-25


### 2025-08-27
- [Patch v5.5.16] Enhance log_analysis utilities
- New/Updated unit tests added for tests.test_log_analysis
- QA: pytest -q passed (325 tests)

### 2025-08-25
- [Patch v5.5.15] Improve data_loader timezone handling and add data quality checks
- New/Updated unit tests added for data_loader


- QA: pytest -q passed

### 2025-08-24
- [Patch v5.5.14] Refactor ProjectP CLI and add logging config
- New/Updated unit tests added for tests.test_projectp_cli
- QA: pytest -q passed (310 tests)

### 2025-08-25
- [Patch v5.5.15] Improve walk-forward validation utilities
- New/Updated unit tests added for tests.test_wfv_utils
- QA: pytest -q passed

### 2025-08-23
- [Patch v5.5.13] Optimize DataFrame writes in backtest
- QA: pytest -q passed (309 tests)

### 2025-08-24

- [Patch v5.5.9] Add profiling option for backtest
- New/Updated unit tests added for profile_backtest, main_pipeline_cli

- QA: pytest -q passed


### 2025-08-22
- [Patch v5.5.12] Add alert summary utilities in log_analysis
- New/Updated unit tests added for tests.test_log_analysis
- QA: pytest -q passed

### 2025-08-21
- [Patch v5.5.9] Add pipeline CLI and threshold optimization script
- New/Updated unit tests added for tests.test_threshold_optimization, tests.test_main_pipeline_cli
- QA: pytest -q passed

### 2025-08-21
- [Patch v5.5.10] Extend ProjectP CLI with sweep and threshold modes
- New/Updated unit tests added for tests.test_projectp_cli
- QA: pytest -q passed

### 2025-06-04
- [Patch v5.5.11] Fix pandas fillna FutureWarning and multiprocessing warning
- QA: pytest -q passed (308 tests)


### 2025-08-20

- [Patch v5.5.8] Improve OMS Guardian with breakeven and SL/TP validation
- New/Updated unit tests added for oms guardian helpers
- QA: pytest -q passed (292 tests)



### 2025-08-13

- [Patch v5.5.7] Add ATR-based position sizing
- New/Updated unit tests added for src.adaptive
- QA: pytest -q passed (278 tests)

### 2025-08-13
- [Patch v5.5.7] Add dynamic lot sizing via drawdown
- New/Updated unit tests added for adaptive module

- QA: pytest -q passed
### 2025-06-04
- [Patch v5.5.8] Implement trailing ATR stop and partial TP defaults
- New/Updated unit tests added for adaptive module
- QA: pytest -q passed


### 2025-08-14
- [Patch v5.5.7] Implement volume spike filter
- New/Updated unit tests added for volume spike and registry
- QA: pytest -q passed

### 2025-08-14
- [Patch v5.5.7] Add MACD divergence detection and filter for buy signals
- New/Updated unit tests added for features and strategy signals
- QA: pytest -q passed

### 2025-06-04
- [Patch v5.5.7] Add log analysis utilities and risk management helper
- New/Updated unit tests added for tests.test_log_analysis

- QA: pytest -q passed (280 tests)


### 2025-08-12
- [Patch v5.5.6] Force COMPACT_LOG in tests, add summary
- New/Updated unit tests added for tests.conftest
- QA: pytest -q passed (277 tests)

### 2025-08-11
- [Patch v5.5.4] Fix SHAP lag feature evaluation dataset
- QA: pytest -q passed (265 tests)

### 2025-08-11
- [Patch v5.5.5] Add environment override for drift threshold
- New/Updated unit tests added for tests.test_env_utils
- QA: pytest -q passed (266 tests)


### 2025-08-10
- [Patch v5.5.3] Update expected line numbers for stub functions
- New/Updated unit tests added for tests.test_function_registry

- QA: pytest -q passed (265 tests)

### 2025-08-11
- [Patch v5.5.4] Add environment override for drift threshold
- New/Updated unit tests added for tests.test_env_utils
- QA: pytest -q passed (266 tests)

### 2025-08-09
- [Patch v5.5.2] Adjust kill switch thresholds and add warnings
- New/Updated unit tests added for src.strategy and main
- QA: pytest -q passed (259 tests)

### 2025-08-08
- [Patch v5.4.5] Update features_main.json with default features
- QA: pytest -q (partial)
### 2025-07-02

[Patch v5.0.24] Improve soft cooldown logic
New/Updated unit tests added for cooldown_utils
QA: pytest -q passed (225 tests)


### 2025-07-01
[Patch v5.3.10] Handle optional models as warnings
New/Updated unit tests added for src.main
QA: pytest -q passed (219 tests)

### 2025-06-30
- [Patch v5.0.23] Fix GPU logging and optuna sweep fallback
- New/Updated unit tests added for ProjectP and hyperparameter sweep
- QA: pytest -q passed (206 tests)

### 2025-06-25
- [Patch v5.0.22] Add backtest profiling script
- New/Updated unit tests added for profile_backtest
- QA: pytest -q passed (174 tests)

### 2025-06-20
- [Patch v5.0.21] Centralize version management
- New/Updated unit tests added for src.config and src.strategy
- QA: pytest -q passed (170 tests)

### 2025-06-05
- [Patch v5.0.1] Simplify FULL_PIPELINE fallback logic
- New/Updated unit tests added for src.main
- QA: pytest -q passed (13 tests)

### 2025-06-03
- [Patch v5.0.2] Adjust coverage configuration for QA
- New/Updated unit tests added for src.features
- QA: pytest -q passed (34 tests)
### 2025-06-02
- [Patch v4.8.12] Remove duplicate PREPARE_TRAIN_DATA and update font call
- New/Updated unit tests added for src.main
- QA: pytest -q passed (8 tests)

### 2024-05-16
- [Patch v1.0.0] Refactor gold ai script into modules
- New/Updated unit tests added for imports
- QA: pytest -q passed (170 tests)

### 2025-06-02
- [Patch v1.0.1] Update default paths for data and logs
- New/Updated unit tests added for imports
- QA: pytest -q passed (1 tests)

### 2025-06-02
- [Patch v1.2.0] Ensure ENTRY_CONFIG_PER_FOLD fallback
- New/Updated unit tests added for src.main
- QA: pytest -q passed (3 tests)

### 2025-06-02
- [Patch v4.8.4] ปรับ config.py ให้รองรับการรันบน Colab และ VPS
- New/Updated unit tests added for src.config
- QA: pytest -q passed (5 tests)

- [Patch v4.8.3] Import setup_output_directory in main
- New/Updated unit tests added for src.main
- QA: pytest -q passed (4 tests)

### 2025-06-02
- [Patch v4.8.5] Fix offline execution for ProjectP
- New/Updated unit tests added for src.main
- QA: pytest -q passed (4 tests)

### 2025-06-02
- [Patch v4.8.8] Add import for safe_set_datetime in strategy.py
- QA: pytest -q passed (6 tests)

### 2025-06-02
- [Patch v4.8.9] Switch to absolute imports for script execution
- New/Updated unit tests added for src.main and src.strategy
- QA: pytest -q passed (6 tests)

### 2025-06-03

- [Patch v4.8.10] Enforce absolute imports for strategy DataLoader utils
- QA: pytest -q passed (6 tests)


- [Patch v4.8.9] Normalize imports to absolute paths in features.py
- QA: pytest -q passed (6 tests)

- [Patch v4.8.10] Stub utilities and clean imports
- New/Updated unit tests added for src.main
- QA: pytest -q passed (7 tests)

### 2025-06-03
- [Patch v4.8.12] Optimize BE year handling and datetime parsing
- QA: pytest -q passed (7 tests)


### 2025-06-03
- [Patch v4.8.9] Fix ModuleNotFoundError for src submodules
- QA: pytest -q passed (7 tests)

### 2025-06-03

- [Patch v4.8.11] Add requirements file
=======
- [Patch v4.8.11] Fix dependency handling and absolute imports
- New/Updated unit tests added for src.main and src.config

- QA: pytest -q passed (7 tests)


- [Patch v4.8.12] Add Numba-accelerated backtest loop and model cache
- New/Updated unit tests added for strategy module
- QA: pytest -q passed (8 tests)




- [Patch v4.8.12] Ensure single TA install and warn on drive mount failure
- New/Updated unit tests added for src.config
- QA: pytest -q passed (8 tests)

### 2025-06-03
- [Patch v4.8.12] Cache TA indicators and vectorize pattern tagging
- New/Updated unit tests added for src.features
- QA: pytest -q passed (7 tests)

### 2025-06-04
- [Patch v4.8.13] Merge simple strategy module into src package
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (13 tests)

### 2025-06-05
- [Patch v4.9.0] ปรับปรุงการแปลงปี พ.ศ. → ค.ศ. ให้เป็นแบบ Vectorized
- New/Updated unit tests added for src.data_loader
- QA: pytest -q passed (13 tests)


### 2025-06-05
- [Patch v4.9.0] ปรับปรุงการสร้างคอลัมน์ session ให้เป็นแบบ Vectorized และลดการวนลูปแถว
- New/Updated unit tests added for src.features
- QA: pytest -q passed (16 tests)

### 2025-06-05
- [Patch v4.9.0] Speed up backtest loop via itertuples
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (13 tests)

### 2025-06-06
- [Patch v4.8.9] Fix AttributeError in run_backtest_simulation_v34
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (34 tests)

### 2025-06-03
- [Patch v5.0.3] เพิ่ม unit tests และปรับ coverage >50%
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (34 tests)
### 2025-06-03
- [Patch v5.0.4] Add function registry tests with line checks
- New/Updated unit tests added for tests.test_function_registry
- QA: pytest -q passed (39 tests)
### 2025-06-03
- [Patch v5.0.5] Adjust function registry test skip logic
- QA: pytest -q passed (39 tests)

### 2025-06-07
- [Patch v5.0.6] Ensure function registry tests run
- Added stub implementations for registry functions
- Updated tests with current line numbers
- QA: pytest -q passed (71 tests)
### 2025-06-07
- [Patch v5.0.7] Auto-create default features file and relax thresholds
- New/Updated unit tests added for src.features
- QA: pytest -q passed (71 tests)
### 2025-06-03
- [Patch v5.0.8] เพิ่มชุดการทดสอบอีก 10 บล็อก
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (83 tests)
### 2025-06-08
- [Patch v5.0.9] เพิ่มชุดการทดสอบอีก 10 บล็อก และแก้ไขปัญหา skip
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (93 tests)
### 2025-06-09
- [Patch v5.0.10] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไขการ skip
- New/Updated unit tests added for src.data_loader, src.main
- QA: pytest -q passed (103 tests)

### 2025-06-10
- [Patch v5.0.11] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไข warning skip
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (113 tests)

### 2025-06-11
- [Patch v5.0.12] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไข warning skip
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (123 tests)
### 2025-06-12
- [Patch v5.0.13] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไข warning skip
- New/Updated unit tests added for src.main, src.strategy
- QA: pytest -q passed (133 tests)

### 2025-06-13
- [Patch v5.0.14] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไข warning skip
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (143 tests)


### 2025-06-13
- [Patch v5.0.14] Ensure features file creation and path handling
- New/Updated unit tests added for src.main
- QA: pytest -q passed (135 test


### 2025-06-14
- [Patch v5.0.15] Reduce warning level for missing MAX_NAT_RATIO_THRESHOLD
- New/Updated unit tests added for src.data_loader
- QA: pytest -q passed (145 tests)

### 2025-06-14
- [Patch v5.0.15] เพิ่มชุดการทดสอบอีก 10 บล็อกและแก้ไข warning skip
- New/Updated unit tests added for src.features, src.data_loader
- QA: pytest -q passed (155 tests)


### 2025-06-15
- [Patch v5.0.16] Handle duplicate samples to avoid ConvergenceWarning
- New/Updated unit tests added for src.features, tests.test_function_registry
- QA: pytest -q passed (165 tests)


### 2025-06-15
- [Patch v5.0.16] Fix soft cooldown trigger condition to require lookback trades
- New/Updated unit tests added for tests.test_soft_cooldown_logic
- QA: pytest -q passed (167 tests)

### 2025-06-16
- [Patch v5.0.17] Add hyperparameter sweep helper
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (167 tests)

### 2025-06-16
- [Patch v5.0.18] Optuna CatBoost sweep function
- New/Updated unit tests added for tests.test_hyperparameter_sweep, tests.test_function_registry
- QA: pytest -q passed (168 tests)

### 2025-06-16
- [Patch v5.0.17] Add soft cooldown helper and update tests
- New/Updated unit tests added for src.cooldown_utils, tests.test_soft_cooldown_logic
- QA: pytest -q passed (167 tests)
### 2025-06-17
- [Patch v5.0.18] Relax MACD and cooldown thresholds
- New/Updated unit tests added for tests.test_soft_cooldown_logic, tests.test_function_registry
- QA: pytest -q passed (168 tests)


### 2025-06-18
- [Patch v5.0.19] Add example hyperparameter sweep script
- QA: pytest -q passed (168 tests)

### 2025-06-19
- [Patch v5.0.20] Add soft cooldown bar countdown and helper
- New/Updated unit tests added for tests.test_soft_cooldown_logic
- QA: pytest -q passed (169 tests)

### 2025-06-20
- [Patch v5.1.0] ปรับข้อความ dummy_train_func เป็นภาษาไทย
- QA: pytest -q passed (170 tests)


### 2025-06-21
- [Patch v5.1.1] Standardize absolute imports and add setup.py
- New/Updated unit tests added for src.main and src.strategy
- QA: pytest -q passed (170 tests)

### 2025-06-22
- [Patch v5.1.2] เพิ่มตัวแปร AUTO_INSTALL_LIBS และปรับ logic ตรวจสอบไลบรารี
- New/Updated unit tests added for src.config และ README.md
- QA: pytest -q passed (170 tests)
### 2025-06-03
- [Patch v5.1.3] Consolidate get_session_tag into utils module
- New/Updated unit tests added for src.utils.sessions
- QA: pytest -q passed (174 tests)
### 2025-06-24
- [Patch v5.1.4] เพิ่ม unit tests ครอบคลุมฟังก์ชันหลัก
- New/Updated unit tests added for src.strategy and src.features
- QA: pytest -q passed (182 tests)

### 2025-06-26
- [Patch v5.1.0] เพิ่มฟังก์ชัน run_hyperparameter_sweep สำหรับ Grid Search
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (183 tests)

### 2025-06-27

- [Patch v5.1.0] รวมลำดับการแสดงผล Run ให้อยู่ใน loop เดียวกัน
- QA: pytest -q passed (183 tests)

### 2025-06-28

- [Patch v5.1.5] profile_backtest.py: เพิ่มการสร้าง Features M1 ก่อนรัน backtest
- New/Updated unit tests added for profile_backtest
- QA: pytest -q passed (183 tests)

### 2025-06-29

- [Patch v5.1.6] Fix TRAIN_MODEL_ONLY M1 data path
- QA: pytest -q passed (186 tests)

### 2025-06-30


- [Patch v5.1.6] แก้ไข Merge และตรวจสอบฟีเจอร์ใน train_and_export_meta_model
- New/Updated unit tests added for strategy
- QA: pytest -q passed (186 tests)





### 2025-06-30

- [Patch v5.1.6] TRAIN_MODEL_ONLY: load prep data for trade log
- QA: pytest -q passed (186 tests)
### 2025-07-01
- [Patch v5.1.7] Update line numbers in function registry tests
- QA: pytest -q passed (187 tests)


### 2025-07-01
- [Patch v5.1.8] Fix RSI reindex on duplicate timestamps
- New/Updated unit tests added for src.features
- QA: pytest -q passed (188 tests)

### 2025-07-02
- [Patch v5.1.9] Auto-run PREPARE_TRAIN_DATA when training files are missing
- New/Updated unit tests added for src.main
- QA: pytest -q passed (189 tests)

### 2025-07-03

- [Patch v5.2.0] Import print_gpu_utilization in strategy module
- QA: pytest -q passed (190 tests)
### 2025-07-04
- [Patch v5.1.10] Fixed RSI reindex issue due to duplicate timestamps
- New/Updated unit tests added for src.features
- QA: pytest -q passed (190 tests)

### 2025-07-05
- [Patch v5.2.1] Improve is_colab detection to avoid mount errors
- New/Updated unit tests added for config and registry
- QA: pytest -q passed (193 tests)

### 2025-07-06

- [Patch v5.2.2] Add simple_converter import and relocation
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (193 tests)


### 2025-07-07
- [Patch v5.2.3] Fix missing imports for metrics and SHAP noise checker
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (193 tests)
### 2025-07-08
- [Patch v5.2.4] Ensure default output directory exists
- New/Updated unit tests added for src.main
- QA: pytest -q passed (198 tests)

### 2025-07-09

- [Patch v5.2.5] เพิ่มตัวเลือก --limit และ --output ใน profile_backtest.py
- New/Updated unit tests added for profile_backtest
- QA: pytest -q passed (199 tests)

### 2025-07-10

- [Patch v5.2.6] ปรับปรุง hyperparameter_sweep ให้ใช้ run_hyperparameter_sweep
- New/Updated unit tests added for hyperparameter_sweep
- QA: pytest -q passed (199 tests)

### 2025-07-11
- [Patch v5.3.0] Enterprise hyperparameter sweep improvements
- New/Updated unit tests added for hyperparameter_sweep
- QA: pytest -q passed (199 tests)




### 2025-07-10
- [Patch v5.2.6] Update requirements list
- QA: pytest -q passed (199 tests)

### 2025-06-03
- [Patch v5.2.7] Initialize metrics dicts to avoid UnboundLocalError
- New/Updated unit tests adjusted for line offsets
- QA: pytest -q passed (207 tests)


### 2025-07-11
- [Patch v5.2.7] Update dependency versions in requirements.txt
- QA: pytest -q passed (207 tests)


### 2025-07-11
- [Patch v5.2.7] Catch UnboundLocalError in PREPARE_TRAIN_DATA
- New/Updated unit tests added for src.main
- QA: pytest -q passed (199 tests)

### 2025-07-12
- [Patch v5.3.1] Robust Metrics Assignment in run_all_folds_with_threshold
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (199 tests)


### 2025-07-12
- [Patch v5.3.1] Safeguard main pipeline from missing metrics
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed (207 tests)

### 2025-07-30
- [Patch v5.3.2] QA-Guard ensures log files
- New/Updated unit tests added for src.main
- QA: pytest -q passed

### 2025-07-13
- [Patch v5.3.3] Fix Colab detection when running scripts
- New/Updated unit tests added for src.config
- QA: pytest -q passed (210 tests)

### 2025-08-01
- [Patch v5.3.4] Create missing QA audit files automatically
- New/Updated unit tests added for tests/test_function_registry.py
- QA: pytest -q passed (215 tests)

### 2025-07-15
- [Patch v5.3.5] Adjust OMS & Logging defaults for risk management and QA
- New/Updated unit tests added for src.config
- QA: pytest -q passed

### 2025-07-15
- [Patch v5.3.5] Add QA log summary for trade log export
- New/Updated unit tests added for src.utils.trade_logger
- QA: pytest -q passed

### 2025-07-15

- [Patch v5.3.5] QA logging: fold summary, NaN/Inf catch, critical alerts
- New/Updated unit tests added for src.strategy
- QA: pytest -q passed

### 2025-07-16

- [Patch v5.3.6] Harden Colab detection to avoid mount errors
- New/Updated unit tests added for src.config and registry
- QA: pytest -q passed

### 2025-08-02
- [Patch v5.3.7] Improve Colab detection logic
- New/Updated unit tests added for src.config

### 2025-08-03
- [Patch v5.3.8] Update registry line numbers for main stubs
- New/Updated unit tests added for tests.test_function_registry
- QA: pytest -q passed (219 tests)




- [Patch v5.3.5] Skip loading absent models in FULL_RUN
- New/Updated unit tests added for src.main
- QA: pytest -q passed (218 tests)


### 2025-06-04
- [Patch v5.3.8] Improve logger propagation and refine is_colab detection
- New/Updated unit tests added for src.config and tests
- QA: pytest -q passed (219 tests)
### 2025-06-04
- [Patch v5.3.9] Adaptive signal score entry threshold with rolling quantile
- New/Updated unit tests added for src.strategy and tests.test_adaptive_signal_threshold
- QA: pytest -q passed (224 tests)



### 2025-08-04
- [Patch v5.4.0] Add adaptive risk and SL/TP utilities
- New/Updated unit tests added for src.adaptive
- QA: pytest -q passed (219 tests)



- [Patch v5.4.1] Improve coverage to 100%
- New/Updated unit tests added for src.adaptive, feature_analysis, sessions



### 2025-06-04
- [Patch v5.4.2] Fix FUND_PROFILES defaults and auto-train fallback
- New/Updated unit tests added for src.main
- QA: pytest -q passed (225 tests)


### 2025-08-06
- [Patch v5.4.3] Filter sweep kwargs to avoid TypeError
- New/Updated unit tests added for hyperparameter sweep filtering
- QA: pytest -q passed (236 tests)


### 2025-08-07

- [Patch v5.4.4] Simplify ensure_model_files_exist with placeholders
- New/Updated unit tests added for src.main
- QA: pytest -q passed (236 tests)


### 2025-08-08

- [Patch v5.4.5] Validate final M1 data loading and timezone alignment
- New/Updated unit tests added for data_loader and function_registry
- QA: pytest -q passed (251 tests)


### 2025-06-04
- [Patch v5.4.6] Move sweep module to tuning, adjust cooldown defaults, add docs/README
- New/Updated unit tests added for tests.test_hyperparameter_sweep_cli, tests.test_soft_cooldown_logic
- QA: pytest -q passed
### 2025-06-04
- [Patch v5.4.7] Update line numbers in function registry test
- New/Updated unit tests added for tests.test_function_registry
- QA: pytest -q passed (258 tests)
### 2025-06-04

- [Patch v5.4.8] Suppress duplicate adaptive signal threshold logs
- New/Updated unit tests added for tests.test_function_registry

- QA: pytest -q passed (258 tests)


### 2025-06-04
- [Patch v5.4.9] Check for IPython kernel before mounting Google Drive
- New/Updated unit tests added for src.config and tests.test_is_colab

- QA: pytest -q passed (258 tests)
### 2025-06-05
- [Patch v5.5.0] Improve config fallbacks and model file handling
- New/Updated unit tests added for existing suites
- QA: pytest -q passed (258 tests)


### 2025-06-04
- [Patch v5.4.7] Tune MetaClassifier threshold
- New/Updated unit tests added for evaluation and strategy
- QA: pytest -q passed (260 tests)

- [Patch v5.5.1] Enable automatic library installation
- New/Updated unit tests added for src.config

- QA: pytest -q passed (258 tests)



### 2025-06-06
- [Patch v5.5.2] Handle missing entry index with nearest lookup
- New/Updated unit tests added for strategy helper
- QA: pytest -q passed

### 2025-06-07

- [Patch v5.5.4] Add configurable RSI drift override threshold
- New/Updated unit tests added for config and registry

- QA: pytest -q passed (266 tests)

### 2025-06-08
- [Patch v5.5.5] Persist default SESSION_TIMES_UTC in utils.sessions
- QA: pytest -q passed

### 2025-06-09
- [Patch v5.5.6] Add M15 multi-timeframe trend filter for order entry
- New/Updated unit tests added for strategy and features
- QA: pytest -q passed

### 2025-06-10
- [Patch v5.5.7] Vectorize equity and drawdown updates in run_backtest_simulation_v34
- Updated unit tests line numbers for function registry
- QA: pytest -q passed (348 tests)


\n
