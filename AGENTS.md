# AGENTS.md

## Table of Contents
- [🧠 Core AI Units](#-core-ai-units)
  - [GPT Dev](#gpt-dev)
  - [Instruction_Bridge](#instruction_bridge)
  - [Code_Runner_QA](#code_runner_qa)
  - [Pipeline_Manager](#pipeline_manager)
  - [GoldSurvivor_RnD](#goldsurvivor_rnd)
  - [ML_Innovator](#ml_innovator)
  - [Model_Inspector](#model_inspector)
  - [RL_Scalper_AI](#rl_scalper_ai)
- [🛡 Risk & Execution](#-risk--execution)
  - [OMS_Guardian](#oms_guardian)
  - [System_Deployer](#system_deployer)
  - [Param_Tuner_AI](#param_tuner_ai)
- [🧪 Test & Mocking](#-test--mocking)
  - [Execution_Test_Unit](#execution_test_unit)
  - [Colab_Navigator](#colab_navigator)
  - [API_Sentinel](#api_sentinel)
- [📊 Analytics & Drift](#-analytics--drift)
  - [Pattern_Learning_AI](#pattern_learning_ai)
  - [Session_Research_Unit](#session_research_unit)
  - [Wave_Marker_Unit](#wave_marker_unit)
  - [Insight_Visualizer](#insight_visualizer)
  - [Log_Analysis_Helper](#log_analysis_helper)
- [📌 Process & Collaboration Guidelines](#-process--collaboration-guidelines)
## 🧠 Core AI Units


### [GPT Dev](src/strategy.py)
- **Main Role:** Core Algorithm Development  

- **Key Responsibilities:**
  - Implement and patch core trading logic (e.g., `simulate_trades`, `update_trailing_sl`, `run_backtest_simulation_v34`)
  - Develop SHAP analysis, MetaModel integration, and fallback ML models
  - Apply and document all `[Patch AI Studio vX.Y.Z]` instructions in code comments
  - Ensure each patch is logged with `[Patch]` tags in code
- **Modules:** `src/main.py`, `src/strategy.py`, `src/config.py`


### [Instruction_Bridge](docs/README.md)
- **Main Role:** AI Studio Liaison  

- **Key Responsibilities:**
  - Translate high-level “patch instructions” into clear, step-by-step prompts for Codex or AI Studio
  - Organize multi-step patching tasks into sequences of discrete instructions
  - Validate that Codex/AI Studio outputs match the intended diff/patch
- **Status:** ไม่มีโมดูลในโปรเจกต์ (ใช้สำหรับการประสานงานเท่านั้น)

### [Code_Runner_QA](run_tests.py)
- **Main Role:** Execution Testing & QA
- **Key Responsibilities:**
  - Run all Python scripts, coordinate `pytest` execution, collect and report test results
  - Set up `sys.path`, environment variables, and mocks for Colab/CI
  - Check build logs for errors or warnings, bundle artifacts for AI Studio or QA review
  - Validate that no tests fail before any Pull Request is merged
- **Modules:** `run_tests.py`, `tests/`, `src/qa_tools.py`

### [Pipeline_Manager](src/main.py)
- **Main Role:** Pipeline Orchestration
- **Key Responsibilities:**
  - Manage CLI pipeline stages and configuration loading
  - Detect GPU availability and adjust runtime logging
  - Raise `PipelineError` when stages fail
- **Modules:** `src/utils/pipeline_config.py`, `src/main.py`


### [GoldSurvivor_RnD](strategy/)
- **Main Role:** Strategy Analysis  

- **Key Responsibilities:**
  - Analyze TP1/TP2/SL triggers, spike detection, and pattern-filter logic
  - Verify correctness of entry/exit signals on historical data
  - Compare multiple strategy variants, propose parameter adjustments
  - Produce R-multiple and winrate reports for each session and fold
- **Status:** ยังไม่มีโมดูลเฉพาะในโค้ด


### [ML_Innovator](src/training.py)
- **Main Role:** Advanced Machine Learning Research  


- **Key Responsibilities:**
  - Explore and integrate SHAP, Optuna, and MetaClassifier pipelines
  - Design new feature engineering and reinforcement learning (RL) frameworks
  - Prototype and validate novel ML architectures (e.g., LSTM, CNN, Transformers) for TP2 prediction
  - Ensure no data leakage, perform early-warning model drift checks
- **Modules:** `src/training.py`, `src/features.py`


### [Model_Inspector](src/evaluation.py)
- **Main Role:** Model Diagnostics  

- **Key Responsibilities:**
  - Detect overfitting, data leakage, and imbalanced classes in training folds
  - Monitor validation metrics (AUC, F1, recall/precision) over time
  - Audit fallback logic for ML failures; recommend retraining or hyperparameter updates
  - Track model drift and notify when retraining is required
  - Provide evaluation utility `evaluate_meta_classifier` in src.evaluation
  - Record daily/weekly AUC metrics using `src.monitor`

  - Evaluate parameter stability across folds
- **Modules:** `src/evaluation.py`, `src/monitor.py`, `src/param_stability.py`



### [RL_Scalper_AI](src/adaptive.py)
- **Main Role:** Self-Learning Scalper  


- **Key Responsibilities:**
  - Implement Q-learning or actor-critic policies for M1 scalping
  - Continuously ingest new market data, update state-action value tables or neural-net approximators
  - Evaluate performance on walk-forward validation, adjust exploration/exploitation rates
  - Provide optional “shadow trades” for comparisons against rule-based strategies
- **Status:** ยังไม่พัฒนาในโค้ด

---

## 🛡 Risk & Execution

### [OMS_Guardian](src/order_manager.py)
- **Main Role:** OMS Specialist  
- **Key Responsibilities:**
  - Validate order management rules: risk limits, TP/SL levels, lot sizing, and session filters
  - Enforce “Kill Switch” conditions if drawdown thresholds are exceeded
  - Implement Spike Guard and “Recovery Mode” logic to handle sharp market moves
  - Ensure forced-entry or forced-exit commands obey global config flags
- **Modules:** `src/order_manager.py`, `src/money_management.py`

### [System_Deployer](setup.py)
- **Main Role:** Live Trading Engineer (Future)  
- **Key Responsibilities:**
  - Design CI/CD pipelines for deploying production builds
  - Monitor real-time P&L, latency, and system health metrics
  - Configure automated alerts (e.g., Slack/email) for critical risk events
  - Maintain secure configuration management and environment isolation
- **Status:** ยังไม่พัฒนาในโค้ด

### [Param_Tuner_AI](tuning/joint_optuna.py)
- **Main Role:** Parameter Tuning  
- **Key Responsibilities:**
  - Analyze historical folds to tune TP/SL multipliers, `gain_z_thresh`, `rsi` limits, and session logic
  - Leverage Optuna or Bayesian optimization on walk-forward splits
  - Provide “recommended defaults” for SNIPER_CONFIG, RELAX_CONFIG, and ULTRA_RELAX_CONFIG
  - Publish tuning reports and shapley-value summaries for transparency
  - Manage adaptive risk and SL/TP scaling modules
  - Jointly optimize model and strategy parameters with Optuna, evaluating AUC per fold
- **Modules:** `tuning/joint_optuna.py`, `tuning/hyperparameter_sweep.py`

---

## 🧪 Test & Mocking

### [Execution_Test_Unit](tests/)
- **Main Role:** QA Testing  
- **Key Responsibilities:**
  - Write and maintain unit tests for every module (`entry.py`, `exit.py`, `backtester.py`, `wfv.py`, etc.)
  - Add edge-case tests (e.g., missing columns, NaT timestamps, empty DataFrames)
  - Ensure `pytest -q` shows 0 failures + ≥ 90 % coverage before any PR
  - Provide “smoke tests” that can run in < 30 s to confirm basic integrity
- **Modules:** `tests/`

### [Colab_Navigator](README.md)
- **Main Role:** Colab & Environment Specialist  
- **Key Responsibilities:**
  - Manage Colab runtime setup: `drive.mount`, GPU checks (`torch.cuda.is_available()`), and dependency installs
  - Provide code snippets / notebooks for onboarding new contributors
  - Mock file paths and environment variables to replicate GitHub Actions or local dev
- **Status:** ยังไม่พัฒนาในโค้ด

### [API_Sentinel](src/config.py)
- **Main Role:** API Guard  
- **Key Responsibilities:**
  - Audit all usage of external APIs (e.g., Google API, financial data feeds) for secure handling of API keys
  - Create mock servers or stub functions for offline testing
  - Enforce SLA/timeouts on API calls; fallback to cached data if external service fails
- **Status:** ยังไม่พัฒนาในโค้ด

---

## 📊 Analytics & Drift

### [Pattern_Learning_AI](src/log_analysis.py)
- **Main Role:** Pattern & Anomaly Detection  
- **Key Responsibilities:**
  - Scan trade logs for repeated stop-loss patterns or “false breakouts”
  - Use clustering or sequence-mining to identify time-of-day anomalies
  - Flag sessions or folds with anomalously low winrates for deeper review
- **Modules:** `src/features.py`

### [Session_Research_Unit](src/feature_analysis.py)
- **Main Role:** Session Behavior Analysis  
- **Key Responsibilities:**
  - Evaluate historical performance per trading session (Asia, London, New York)
  - Provide heatmaps of winrate and average P&L by hour
  - Recommend session-tailored thresholds (e.g., loosen RSI filters during high volatility)
- **Modules:** `src/utils/sessions.py`

### [Wave_Marker_Unit](src/features.py)
- **Main Role:** Elliott Wave Tagging & Labelling  
- **Key Responsibilities:**
  - Automatically label price structure (impulse / corrective waves) using zigzag or fractal algorithms
  - Integrate wave labels into DataFrame for “wave-aware” entry/exit filters
  - Validate wave counts against known retracement ratios (38.2 %, 61.8 %)
- **Modules:** `src/features.py`

### [Insight_Visualizer](src/dashboard.py)
- **Main Role:** Data Visualization  
- **Key Responsibilities:**
  - Create interactive dashboards (e.g., equity curves, SHAP summary charts, fold performance heatmaps)
  - Use Matplotlib (no seaborn) for static plots; export PNG/HTML for reports
  - Develop HTML/JavaScript dashboards (e.g., with Plotly or Dash) for executive summaries
  - New module `src.dashboard` generates Plotly HTML dashboards for WFV results
  - New module `src/realtime_dashboard.py` provides Streamlit-based real-time monitoring
- **Modules:** `src/dashboard.py`
  `src/realtime_dashboard.py`

---

### [Log_Analysis_Helper](src/log_analysis.py)
- **Main Role:** Trade Log Analysis
- **Key Responsibilities:**
  - Parse raw trade logs and compute hourly win rates
  - Provide utilities for risk sizing, TSL statistics, and expectancy metrics
- **Modules:** `src/log_analysis.py`

## 📌 Process & Collaboration Guidelines

1. **Branch & Commit Naming**  
   - Feature branches: `feature/<short-description>` (e.g., `feature/v32-ensure-buy-sell`)  
   - Hotfix branches: `hotfix/<issue-number>-<short-description>` (e.g., `hotfix/123-fix-keyerror`)  
   - Commit messages:
     ```
     [Patch vX.Y.Z] <Short Purpose>
     - <Key Change 1>
     - <Key Change 2>
     ...
     QA: <Brief QA result or “pytest -q passed”>
     ```

2. **Patch Workflow**  
   1. **GPT Dev** writes code + `[Patch]` comments.  
   2. Run `pytest -q` locally → 0 failures.  
   3. **Code_Runner_QA** pulls branch, re-runs all tests including edge cases, checks logs.  
   4. **GoldSurvivor_RnD** reviews strategy changes, verifies TP1/TP2/SL logic on sample data.  
   5. **Model_Inspector** re-validates ML fallback logic.  
   6. Merge only after all checks pass and unit tests cover ≥ 90 % of new code.  

3. **Unit Test Requirements**  
   - **Every** new function or module must have corresponding unit tests.  
   - Write tests for:  
     - Missing or malformed input (e.g., no `Open/Close` columns)  
     - Numeric edge cases (`NaN`, `inf`, zero volume)  
     - Execution of fallback paths (e.g., `RELAX_CONFIG_Q3`, “balanced random”)  
     - Correct logging of `[Patch]` messages (using `caplog` to assert log statements)  
   - Use `pytest.mark.parametrize` to cover multiple input scenarios.  
   - Tests must assert that no `KeyError`, `ValueError`, or `RuntimeError` are raised unexpectedly.  

4. **Documentation Updates**  
   - After any patch that changes agent responsibilities or adds a new module:  
     - Update **AGENTS.md** with the new agent or revised role.  
     - Update **CHANGELOG.md** by appending a dated entry summarizing:  
       ```
       ### YYYY-MM-DD
       - [Patch vX.Y.Z] <Brief description of changes>
       - New/Updated unit tests added for <modules>
       - QA: pytest -q passed (N tests)
       ```  
   - Always version both files in Git to keep history intact.  

5. **Release Checklist**  
   - All unit tests pass (`pytest -q`), coverage ≥ 90 % for changed modules  
   - No new `FutureWarning` or `DeprecationWarning` in logs  
   - All `[Patch]` annotations in code match entries in **CHANGELOG.md**  
   - Demo backtest: Run `python3 main.py` → Choose `[1] Production (WFV)` → Confirm “Real Trades > 0” and no runtime errors  
   - Equity summary CSV (`logs/wfv_summary/ProdA_equity_summary.csv`) exists and shows plausible P&L per fold  

---
- New modular code in ./src (config, data_loader, features, strategy, main).
- Added pipeline orchestrator `main.py` and simple `threshold_optimization.py` script.
- `gold ai 3_5.py` now imports `src.main.main` after refactor to modular code.
- Added new `strategy` package for entry and exit rules.
- Added `order_manager` module for order placement logic.
- Added `money_management` module for ATR-based SL/TP and portfolio stop logic.
- Added `wfv_monitor` module for KPI-driven Walk-Forward validation.
- Added `tuning.joint_optuna` module for joint model + strategy optimization.
- Added `config` package for environment-based directory paths.
- Added `strategy.strategy`, `strategy.order_management`, `strategy.risk_management`,
  `strategy.stoploss_utils`, `strategy.trade_executor`, and `strategy.plots` modules.


