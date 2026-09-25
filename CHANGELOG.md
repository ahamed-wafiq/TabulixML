# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-25

### DataCraft v1.0.0 General Availability Release
First official production release of DataCraft: the lightweight, CPU-friendly, tabular machine learning toolkit covering the complete lifecycle from messy data to production inference.

### Core Modules & Capabilities

#### 1. AutoClean -- Automated Data Cleaning & Auditing
- **Non-destructive cleaning**: Original DataFrames are strictly preserved and never mutated in-place.
- **Missing-value imputation**: Strategies (`mean`, `median`, `mode`, `auto`) based on detected column data types.
- **Duplicate handling**: Deterministic duplicate row identification and configurable removal.
- **Column hygiene**: Removal of completely empty columns, normalization to `snake_case`.
- **Quality auditing**: Intelligent warnings for potential ID columns, high-cardinality categories, whitespace/casing inconsistencies, and pairwise feature redundancy.
- **Lifecycle API**: `inspect()`, `preview()`, `clean()`, `report()`, and `history()`.

#### 2. AutoEDA -- Exploratory Data Analysis & Reporting
- **Read-only analysis**: Fast tabular diagnostics without modifying input data.
- **Descriptive statistics**: Comprehensive summaries for both numerical and categorical distributions.
- **Correlation analysis**: Pearson pairwise correlations with configurable threshold filtering.
- **Visualizations**: Automatic, clean matplotlib/seaborn plots (histograms, boxplots, categorical frequency bar charts, correlation heatmaps) with no UI popups during testing.
- **Report export**: Standalone, interactive HTML report generation via `save_report()`.
- **Lifecycle API**: `inspect()`, `summary()`, `correlations()`, `quality()`, `visualize()`, and `save_report()`.

#### 3. AutoPrep -- Leak-Free Preprocessing Pipelines
- **Type detection**: Automatic identification of numerical, categorical, and datetime columns.
- **Zero data leakage**: Preprocessing transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fit strictly on training splits.
- **Deterministic train/test splits**: Reproducible splitting with configurable `test_size` and `random_state`.
- **scikit-learn integration**: Exports full `ColumnTransformer` pipelines compatible with standard ML workflows.
- **Inference transformation**: Leak-free transformation of new/test data via `transform()`.
- **Lifecycle API**: `inspect()`, `preview()`, `split()`, `prepare()`, `get_pipeline()`, and `transform()`.

#### 4. AutoML -- Model Selection, Tuning & Production Persistence
- **Task detection**: Automatic classification vs. regression inference based on target distribution and cardinality.
- **Baseline models**:
  - Classification: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
  - Regression: `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`.
- **Cross-validation**: Stratified K-Fold for classification (accuracy, precision, recall, F1) and K-Fold for regression (MAE, RMSE, R²).
- **Leak-free tuning**: Integrated `RandomizedSearchCV` where preprocessing is fit strictly inside cross-validation folds.
- **Model persistence**: Complete self-contained artifact serialization using `joblib` (`save_model()`, `load_model()`).
- **Production scoring**: Direct raw-DataFrame inference (`predict()`, `predict_proba()`) and rich metadata inspection (`model_info()`).
- **Lifecycle API**: `inspect()`, `detect_task()`, `preview()`, `fit()`, `evaluate()`, `compare()`, `tune()`, `tuning_results()`, `best_models()`, `predict()`, `predict_proba()`, `evaluate_tuned()`, `save_model()`, `load_model()`, `model_info()`.

### Reliability & Consistency
- Standardized public method signatures returning useful Python objects (dictionaries, DataFrames, NumPy arrays) alongside terminal outputs.
- Comprehensive dataset validations: guards against empty DataFrames, single-row data, missing targets, uniform targets, zero features, and all-NaN inputs.
- Explicit lifecycle guards raising clean `RuntimeError` exceptions for out-of-order calls (e.g. predicting before fitting).
- Full 100% test coverage with 357 passing unit and integration tests.

---

## [0.1.0] - 2026-09-24

### Added
- Initial DataCraft release
- AutoClean tabular data cleaning pipeline
- Missing-value detection and imputation (mean, median, mode, auto strategies)
- Duplicate row detection and removal (configurable via `remove_duplicates`)
- Empty column detection and automated removal
- Constant column detection and reporting (flagged only, never deleted)
- Numerical outlier detection using IQR fences (flagged only, never deleted)
- Column name normalization and cleaning (`snake_case` conversion)
- Intelligent data-quality warnings:
  - Possible ID columns (unique values close to row count)
  - High-cardinality categorical columns
  - Inconsistent categorical values (casing and whitespace variants with suggestions)
  - Redundant numerical features (high pairwise Pearson correlation)
  - Optional class imbalance and target leakage analysis
- Interactive preview workflow (`preview()` and `clean(approve=False)`)
- Detailed reporting (`report()`) with before/after comparisons and change logs
- Structured audit history (`history()`) recording operation, column, affected counts, and strategy
- AutoEDA exploratory data analysis module (strictly read-only)
  - `inspect()`: dataset dimensions, data types, missing counts/percentages, unique counts, duplicates
  - `summary()`: descriptive statistics for numerical (count, mean, median, std, min, max) and categorical (unique, top, freq)
  - `correlations()`: numerical Pearson correlation matrix and high correlation pair detection (|r| >= threshold)
  - `quality()`: consolidated data-quality audit (missing rates, duplicates, constant columns, high cardinality, outliers)
  - `report()`: clean human-readable terminal EDA report
- Reset method (`reset()`) for re-analyzing or re-cleaning from scratch
- Immutability guarantees across all operations
- Full test suite with 218 unit, real-world scenario, and AutoEDA tests
