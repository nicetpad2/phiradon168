### 2025-06-05
- [Patch v5.0.1] Simplify FULL_PIPELINE fallback logic
- New/Updated unit tests added for src.main
- QA: pytest -q passed (13 tests)
### 2025-06-02
- [Patch v4.8.12] Remove duplicate PREPARE_TRAIN_DATA and update font call
- New/Updated unit tests added for src.main
- QA: pytest -q passed (8 tests)

### 2024-05-16
- [Patch v1.0.0] Refactor gold ai script into modules
- New/Updated unit tests added for imports
- QA: pytest -q passed (0 tests)

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
