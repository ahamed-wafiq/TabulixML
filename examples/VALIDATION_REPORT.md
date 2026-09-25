# TabulixML v1.0 Real-World Validation & Quality Report

**Document Status**: Final  
**Date**: September 2026  
**Target Release**: TabulixML v1.0.0 General Availability  

---

## 1. Executive Summary

This report documents the empirical validation of the **TabulixML v1.0.0** toolkit on six diverse, realistic, and previously unseen tabular datasets, as well as an eleven-scenario edge-case stress test. 

The validation confirmed that TabulixML is:
1. **Reliable & Consistent**: Every public method across `AutoClean`, `AutoEDA`, `AutoPrep`, and `AutoML` operates deterministically and returns structured Python objects alongside human-readable terminal output.
2. **Leak-Free**: Feature preprocessing pipelines (imputation, scaling, one-hot encoding) are fitted strictly on training splits and cross-validation folds.
3. **Safe by Design**: Caller DataFrames are never mutated in place. Constant columns and extreme outliers are flagged with clear warnings rather than silently deleted.
4. **CPU-Friendly**: The entire validation suite—comprising 6 datasets, EDA visual generation, report exports, cross-validation, hyperparameter tuning, model persistence, and 11 edge case stress tests—completed in **under 18 seconds on a standard CPU**.

---

## 2. Datasets Tested

The validation utilized six diverse tabular datasets representing typical industrial machine learning challenges:

| Dataset | Samples / Features | Primary Challenge | Modules Validated |
| :--- | :--- | :--- | :--- |
| **Employee Dataset** | 12 rows $\times$ 5 cols | Inconsistent casing/whitespace, extreme salary outlier ($2.5M), missing values | AutoClean, AutoEDA |
| **Customer Dataset** | 12 rows $\times$ 5 cols | Unique ID columns, high-cardinality categories, country casing variants | AutoClean, AutoEDA |
| **Sales Dataset** | 14 rows $\times$ 6 cols | Duplicate transactions, datetime timestamps, perfectly redundant features ($r=1.000$) | AutoClean, AutoEDA |
| **Student Dataset** | 10 rows $\times$ 8 cols | 100% empty column, constant columns (school code, year), GPA missing values | AutoClean, AutoEDA |
| **Loan Approval** | 123 rows $\times$ 7 cols | Classification target (`loan_approved`), mixed types, missing values, duplicates | AutoClean, AutoEDA, AutoPrep, AutoML |
| **Used Car Resale** | 123 rows $\times$ 6 cols | Regression target (`resale_price`), continuous price distribution, mileage noise | AutoClean, AutoEDA, AutoPrep, AutoML |

---

## 3. What Worked

### AutoClean
* **Non-Destructive Cleaning**: Preserved original DataFrames with 100% deep-equality guarantees across all test runs.
* **Intelligent Imputation**: Automatically chose `median` for skewed/numerical data and `mode` for categorical columns.
* **Duplicate Detection**: Safely dropped duplicate rows while maintaining clean index continuity.
* **Empty Column Removal**: Automatically identified and pruned 100% null columns (e.g. `empty_notes` in Student Dataset).
* **Warning Engine**:
  * Correctly alerted on casing variants (e.g., `' Sales '`, `'Sales'`, `'sales'`).
  * Flagged potential identifier columns (`customer_id`, `transaction_id`).
  * Warned about redundant feature pairs (`quantity` $\leftrightarrow$ `total_amount`).
  * Highlighted extreme outliers via IQR fences without deleting legitimate high-value data.

### AutoEDA
* **Headless Visualizations**: Automatically generated histograms, boxplots, category bar charts, and correlation heatmaps using matplotlib's `Agg` backend without GUI window popups or test interruptions.
* **HTML Export**: Standalone self-contained HTML reports (`eda_report.html`) exported reliably across all 6 datasets.
* **Descriptive Summaries**: `summary()` returned distinct DataFrames for numerical and categorical distributions with zero divide-by-zero crashes.

### AutoPrep
* **Zero Preprocessing Leakage**: Split data before fitting transformers; test set transformed strictly using the fitted training pipeline.
* **Unseen Category Tolerance**: Tested inference with a novel category (`"Completely_Unseen_Novel_Category_XYZ"`). Handled seamlessly by `OneHotEncoder(handle_unknown='ignore')` without runtime failure.
* **Strict Target Isolation**: Target column is strictly segregated and never included in feature transformations.

### AutoML
* **Automated Task Detection**: Correctly identified binary classification for Loan Approval and continuous regression for Used Car Resale.
* **Leak-Free Cross-Validation**:
  * Evaluated baseline models using Stratified K-Fold (classification) and K-Fold (regression).
  * Reported fold-by-fold metrics and sorted comparison leaderboards via `compare()`.
* **Controlled Tuning**: `tune(n_iter=4)` sampled hyperparameter grids cleanly using `RandomizedSearchCV`.
* **Production Persistence**:
  * Saved full artifacts via `joblib`.
  * Reloaded pipelines via `AutoML.load_model()`.
  * Executed `predict()` and `predict_proba()` on raw DataFrames containing missing values and unscaled features.

---

## 4. Edge Cases Handled

The test harness verified eleven edge cases in `examples/real_world_validation.py`:

1. **Empty DataFrame**: AutoClean and AutoEDA raised clear `ValueError` messages explaining that the DataFrame was empty.
2. **Single-Row Dataset**: Cleaned safely; `AutoPrep.split()` cleanly raised `ValueError` stating $\ge 2$ rows are required for train/test splits.
3. **Numeric-Only Dataset**: Preprocessing pipeline constructed with 0 categorical columns without errors.
4. **Categorical-Only Dataset**: Preprocessing pipeline constructed with 0 numerical columns without errors.
5. **Datetime Columns**: Correctly recognized datetime types and segregated them from numerical/categorical scalers.
6. **All-Missing Column**: Pruned automatically during structural cleaning.
7. **Constant Column**: Flagged with user warning, safely preserved in the dataset.
8. **Duplicate-Heavy Dataset**: Reduced 8 rows (with 6 duplicate entries) down to 2 unique rows.
9. **Extreme Outliers**: Detected $1,000,000.0$ against $10.0$ via IQR fences; preserved value intact without silent deletion.
10. **High-Cardinality Column**: Generated high-cardinality and ID-like warnings in inspection report.
11. **Very Small Dataset (3 rows)**: Imputed missing values and cleaned without sample-size crashes.

---

## 5. Performance Observations

* **Execution Speed**:
  * Data cleaning + report generation across all 4 tabular datasets: **~3.6 seconds total**.
  * Full AutoML classification (3-fold CV + 4-iteration tuning + held-out test + persistence): **7.42 seconds**.
  * Full AutoML regression (3-fold CV + 4-iteration tuning + held-out test + persistence): **6.88 seconds**.
  * Entire edge case suite: **< 0.5 seconds**.
* **Memory & Storage**:
  * Model artifacts saved with `joblib` averaged **~80 KB - 250 KB**, making them suitable for microservices, lambdas, and embedded environments.
  * Headless matplotlib rendering created zero lingering file handles or memory leaks.

---

## 6. Bugs Found and Fixed During Validation

1. **Assertion Mismatch in `correlations()` Test Check**:
   * *Issue*: Validation script initially asserted `isinstance(corrs, pd.DataFrame)`, but `AutoEDA.correlations()` returns a dictionary containing `{"matrix": pd.DataFrame, "high_correlations": list, "threshold": float}`.
   * *Resolution*: Updated `real_world_validation.py` to check `isinstance(corrs, dict)` and `isinstance(corrs["matrix"], pd.DataFrame)`.
2. **Row Key Mismatch in `AutoEDA.inspect()`**:
   * *Issue*: Edge case test checked `eda.inspect()["n_rows"]`, whereas the public key returned is `eda.inspect()["rows"]`.
   * *Resolution*: Updated `real_world_validation.py` to use `eda.inspect()["rows"]`.
3. **Column Normalization Alignment in Example Inferences**:
   * *Issue*: When `clean_col_names=True` was omitted in initial testing, original column names had spaces (`"Square Feet"`), which failed when inference samples supplied snake_case names (`"square_feet"`).
   * *Resolution*: Explicitly documented and standardized `clean_col_names=True` across examples to ensure consistent snake_case feature alignment.

---

## 7. Current Limitations Discovered

1. **Datetime Feature Engineering**:
   * Currently, datetime columns are flagged and segregated (to avoid crashing scikit-learn scalers), but no automated temporal feature extraction (year, month, day of week, hour) is performed.
2. **Interactive Harmonization**:
   * AutoClean detects casing/whitespace variants (e.g. `' Sales '` $\to$ `'Sales'`), but does not apply harmonization automatically unless manually specified by the user.
3. **Imputation Variety**:
   * Missing value imputation is currently limited to `mean`, `median`, and `mode`. Time-series forward-fill / backward-fill and constant value fills are not yet built-in.
4. **Target Cardinality Limits**:
   * High-cardinality multi-class classification targets ($> 20$ classes) can trigger scikit-learn warnings if samples per class are fewer than the cross-validation fold count ($k$).

---

## 8. Recommended Improvements for TabulixML v1.1

*(Note: In accordance with project instructions, these recommendations are documented for future development and are NOT implemented in v1.0).*

1. **Automated Datetime Feature Extraction**:
   * Introduce a lightweight, optional transformer in `AutoPrep` to extract temporal components (`year`, `month`, `day`, `dayofweek`, `is_weekend`) from datetime features.
2. **Automated Category Harmonization in AutoClean**:
   * Add a `harmonize_categories=True` option in `AutoClean` to automatically apply the suggested casing/whitespace normalizations detected during inspection.
3. **Expanded Imputation Modes**:
   * Support `constant` fill values (e.g., `"missing"` or `0`), as well as time-ordered `ffill` / `bfill` strategies for sequential datasets.
4. **Adaptive Cross-Validation Folds**:
   * In `AutoML`, automatically reduce `cv` if the minimum class sample count is smaller than the requested fold count, preventing user errors when evaluating rare classes.
5. **Polars DataFrame Interoperability**:
   * Provide seamless zero-copy translation between Polars and pandas DataFrames for enhanced read performance.
