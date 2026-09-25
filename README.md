# DataCraft

> **Lightweight, transparent, and deterministic tabular data cleaning for Machine Learning -- built with pure Python, pandas, numpy, and scipy.**

[![PyPI version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://pypi.org/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-357%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Introduction

**DataCraft** is a lightweight Python toolkit designed to streamline the critical first mile of tabular machine learning workflows: data cleaning, quality auditing, and exploratory data analysis.

```text
DataCraft
│
├── AutoClean
│   └── Clean the data
│
├── AutoEDA
│   └── Understand the data
│
├── AutoPrep
│   └── Prepare data for ML (leak-free split & preprocessing)
│
└── AutoML
    └── Baseline model training & evaluation
```

Unlike opaque AutoML libraries, DataCraft is **transparent and non-destructive**:
- **Immutability First**: It *never* modifies your original DataFrame in place; `clean()` always returns a clean copy, and `AutoEDA` is strictly read-only.
- **Safety by Design**: It never automatically deletes outliers or drops ambiguous columns without consent.
- **Explainability**: Every action is proposed upfront via `preview()`, audited in `report()`, and logged step-by-step in `history()`.

---

## Problem It Solves

Real-world tabular datasets are notoriously messy—plagued with missing values, duplicate rows, casing and whitespace discrepancies, extreme outliers, constant features, and subtle target leakage.

Data practitioners often face two unfavorable extremes:
1. **Manual Boilerplate**: Writing repetitive, error-prone code for type inference, mode/median calculations, duplicate checks, and outlier flagging for every new dataset.
2. **Opaque AutoML Tools**: Heavy, black-box libraries that alter datasets silently, drop rows unexpectedly, create hard-to-debug side effects, or pull in hundreds of heavy dependencies.

**DataCraft bridges this gap.** It gives you:
- An immediate, comprehensive diagnosis of your data quality via `inspect()`.
- An upfront, safe look at what will change before it happens via `preview()`.
- Deterministic, configurable cleaning that strictly preserves the original data via `clean()`.
- A complete, step-by-step audit trail via `report()` and `history()`.

---

## Current Features

- **Automated Structural Cleaning**:
  - Imputes numerical missing values using `median` or `mean` (or skewness-aware `auto`).
  - Imputes categorical missing values using `mode`.
  - Removes duplicate rows (configurable via `remove_duplicates=True/False`).
  - Drops completely empty columns (100% missing values).
  - Optional column-name normalization (strip, lowercase, slugify, deduplicate).
- **Intelligent Data-Quality Auditing (Flagged only, never modified automatically)**:
  - **ID-like Columns**: Detects primary keys / identifiers with high unique value ratios ($\ge 90\%$).
  - **High-Cardinality Columns**: Identifies categorical features with unusually many distinct categories ($\ge 10$).
  - **Inconsistent Categorical Values**: Flags casing and whitespace discrepancies (e.g., `"Mumbai"`, `"mumbai"`, `" Mumbai "`) and provides suggested normalized replacements.
  - **Redundant Numerical Features**: Identifies highly correlated feature pairs ($|r| \ge 0.90$).
  - **Outlier Flagging**: Flags numerical outliers using standard IQR fences ($Q1 - 1.5 \times \text{IQR}$, $Q3 + 1.5 \times \text{IQR}$); **never deletes rows**.
  - **Target Auditing**: Audits class distribution, warns about class imbalance ($\ge 70\%$ majority), and detects possible target leakage (direct duplicates, near-perfect correlation, 1:1 proxies).

---

## Installation

```bash
# Clone the repository and install in editable mode
pip install -e .

# Or install with developer test dependencies
pip install -e ".[dev]"
```

**Requirements**:
- Python $\ge$ 3.9
- pandas $\ge$ 1.5
- numpy $\ge$ 1.23
- scipy $\ge$ 1.9
- scikit-learn $\ge$ 1.0
- matplotlib $\ge$ 3.5
- seaborn $\ge$ 0.12

---

## Quick Start & Examples

### Basic AutoClean Example

```python
import pandas as pd
from datacraft import AutoClean

df = pd.DataFrame({
    "Full Name": ["Alice", "Bob", "Charlie", "alice", "Alice"],
    "Age": [25.0, 30.0, None, 45.0, 25.0],
    "Salary": [50000.0, 60000.0, 55000.0, 999.0, 50000.0],
    "Department": ["Engineering", "Sales", "HR", "engineering", "Engineering"],
    "Empty Col": [None, None, None, None, None],
})

# Initialize cleaner
cleaner = AutoClean(df)

# 1. Preview planned actions
cleaner.preview()

# 2. Perform cleaning (returns a new DataFrame)
cleaned_df = cleaner.clean()

# 3. View before/after report
cleaner.report()
```

---

### Configuration Options

Customize cleaning strategies to match your ML pipeline needs:

```python
cleaner = AutoClean(
    df,
    numerical_strategy="median",   # 'median', 'mean', or 'auto'
    categorical_strategy="mode",     # 'mode'
    remove_duplicates=True,          # True (default) or False to retain duplicates
    clean_col_names=True,            # True to convert column headers to clean snake_case
    iqr_threshold=1.5,               # Multiplier for IQR outlier fence (default 1.5)
    target="label"                   # Optional target column for leakage & imbalance checks
)
```

---

### `inspect()`

Return a detailed dictionary summarizing data types, missing values, duplicates, outliers, and quality findings:

```python
summary = cleaner.inspect(target="label")

print("Shape:", summary["shape"])
print("Missing values:", summary["missing"])
print("ID-like columns:", summary["id_like_cols"])
print("High-cardinality columns:", summary["high_cardinality_cols"])
print("Inconsistent categories:", summary["inconsistent_categories"])
print("Redundant numerical pairs:", summary["redundant_numerical"])
print("Class imbalance:", summary["class_imbalance"])
print("Target leakage:", summary["target_leakage"])
```

---

### `preview()`

Review proposed transformations and warnings **before** applying any changes:

```python
plan = cleaner.preview()
```

```text
============================================================
  DataCraft -- AutoClean Preview
  (no changes applied yet)
============================================================
  DataFrame : 5 rows x 5 columns
------------------------------------------------------------
  [DROP]   1 completely empty column(s):
      'Empty Col'
  [DROP]   1 duplicate row(s) will be removed.
  [IMPUTE] 1 column(s) have missing values:
      Age                          [  numerical]  1 missing  strategy=median  fill=30
  [WARN]   2 column(s) have inconsistent categorical values:
      'Full Name' variants: ['Alice', 'alice']  ->  suggested: 'Alice'
      'Department' variants: ['Engineering', 'engineering']  ->  suggested: 'Engineering'
  [WARN]   Outliers detected (IQR - NOT removed automatically):
      Salary                          1 outlier(s)  fence [4.81e+04, 6.41e+04]
============================================================
```

---

### `clean()`

Execute cleaning steps. Returns a new DataFrame instance; the original input is guaranteed untouched:

```python
# Apply cleaning
cleaned_df = cleaner.clean()

# Or use approval gate (preview only, returns None):
cleaner.clean(approve=False)
```

---

### `report()`

Print a formatted comparison table showing row/column deltas, exact imputations, dropped columns, and quality warnings:

```python
cleaner.report()
```

```text
============================================================
  DataCraft -- AutoClean Report
============================================================
                  Rows    Cols
  BEFORE             5       5
  AFTER              4       4
  DELTA             -1      -1

  Column types   : {'categorical': 2, 'numerical': 2}

  Changes Made by clean():
  * Dropped 1 empty column(s): ['Empty Col']
  * Removed 1 duplicate row(s).
  * Imputed missing values in 1 column(s):
      Age                          [numerical [median]]  1 value(s) -> 30

------------------------------------------------------------
  Inconsistent categorical values (flagged only, NOT changed):
    * 'Full Name' variants: ['Alice', 'alice'] -> suggested: 'Alice'
    * 'Department' variants: ['Engineering', 'engineering'] -> suggested: 'Engineering'
  Outliers (IQR - flagged only, NOT removed):
    Salary                             1 outlier(s)  fence [4.81e+04, 6.41e+04]
============================================================
```

---

### `history()`

Retrieve an audit trail of every operation, column affected, count of modified values/rows, and strategy used:

```python
for entry in cleaner.history():
    print(f"[{entry['operation']}] col={entry['column']} affected={entry['affected_count']} strategy={entry['strategy']}")
    print(f"  Message: {entry['message']}")
```

---

### `reset()`

Clear the history log and reset the cleaner's internal state to re-analyze or re-clean the original DataFrame:

```python
cleaner.clean()
cleaner.reset()

assert cleaner.history() == []
# Ready to re-inspect or re-clean fresh
cleaner.inspect()
```

---

## AutoEDA (Exploratory Data Analysis)

`AutoEDA` provides fast, read-only exploratory data analysis for tabular datasets without modifying your data.

```python
from datacraft import AutoEDA

eda = AutoEDA(df)
```

### AutoEDA Methods

| Method | Description | Return Type |
| :--- | :--- | :--- |
| `eda.inspect()` | Dimensions, column names, detected data types, missing counts/percentages, unique counts, duplicate count. | `dict` |
| `eda.summary()` | Statistical summaries: numerical (count, mean, median, std, min, max) and categorical (unique, top, freq). | `dict[str, pd.DataFrame]` |
| `eda.summary(column)` | Statistical summary for a single named column. | `pd.Series` |
| `eda.correlations()` | Pearson correlation matrix and identification of highly correlated feature pairs (`\|r\| >= threshold`). | `dict` |
| `eda.quality()` | Consolidated quality audit: missing-value rates, duplicate rows, constant columns, high cardinality, outliers. | `dict` |
| `eda.report()` | Clean, human-readable terminal EDA report combining structural, statistical, and quality insights. | `str` |
| `eda.visualize()` | Automatically generates histograms, boxplots, category frequency bars, and correlation heatmap. | `dict` |
| `eda.visualize(output_dir)` | Saves generated plot PNG images into specified output folder. | `dict` |
| `eda.save_report("eda_report.html")` | Exports a standalone, self-contained HTML report with tables, warnings, and embedded visualizations. | `str` |

### AutoEDA Example

```python
from datacraft import AutoEDA

eda = AutoEDA(df)
eda.inspect()
eda.summary()
eda.visualize()
eda.save_report("eda_report.html")
```

#### Detailed Exploration Example

```python
import pandas as pd
from datacraft import AutoEDA

df = pd.DataFrame({
    "Age": [25.0, 30.0, None, 45.0, 25.0],
    "Salary": [50000.0, 60000.0, 55000.0, 2500000.0, 50000.0],
    "Department": ["Engineering", "Sales", "HR", "Sales", "Engineering"],
})

eda = AutoEDA(df)

# 1. Inspect structural properties
info = eda.inspect()
print("Shape:", info["shape"])
print("Duplicate rows:", info["duplicates"])

# 2. Detailed statistical summaries
stats = eda.summary()
print(stats["numerical"])
print(stats["categorical"])

# 3. Correlation analysis
corrs = eda.correlations(threshold=0.85)
print(corrs["high_correlations"])

# 4. Data-quality audit
qual = eda.quality()
print("Outliers:", qual["outlier_counts"])
print("Missing:", qual["missing_percentage"])

# 5. Full terminal report
eda.report()

# 6. Automatic visualizations (with optional output folder for PNGs)
eda.visualize(output_dir="eda_output")

# 7. Standalone HTML report (self-contained, opens in any browser offline)
eda.save_report("eda_report.html")
```

---

## AutoPrep (Leak-Free ML Preprocessing)

`AutoPrep` prepares cleaned tabular data for machine learning models using reproducible, leak-free scikit-learn pipelines.

```python
from datacraft import AutoPrep

prep = AutoPrep(
    df,
    target="target_column",
    test_size=0.2,
    random_state=42
)
```

### AutoPrep Methods

| Method | Description | Return Type |
| :--- | :--- | :--- |
| `prep.inspect()` | Identifies feature columns, target column, data types, missing counts, and unique value counts. | `dict` |
| `prep.preview()` | Displays proposed numerical, categorical, and datetime preprocessing steps upfront. | `dict` |
| `prep.split()` | Splits dataset into train and test sets using `train_test_split`. | `tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` |
| `prep.prepare()` | Splits data, fits preprocessing **strictly on training data**, transforms train and test. | `tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]` |
| `prep.get_pipeline()` | Returns the fitted scikit-learn `ColumnTransformer` preprocessing pipeline. | `ColumnTransformer` |
| `prep.transform(new_data)` | Transforms new/unseen data using the fitted pipeline without refitting. | `np.ndarray` |
| `prep.get_feature_names()` | Returns preserved feature column names after encoding. | `list[str]` |

### AutoPrep Features & Guarantees

- **Zero Target Leakage**: Preprocessing pipelines are fitted **only on X_train**. The test set and whole dataset are never seen during fitting.
- **Reproducible Train/Test Splits**: Uses scikit-learn's `train_test_split` with configurable `test_size` and `random_state`.
- **Numerical Pipeline**: Imputes missing values with `median` via `SimpleImputer` and standardizes with `StandardScaler` (`scale=True/False`).
- **Categorical Pipeline**: Imputes missing values with `mode` (`most_frequent`) and one-hot encodes via `OneHotEncoder(handle_unknown="ignore")`.
- **Preserved Feature Names**: Employs `verbose_feature_names_out=False` to retain clean, readable feature names.
- **Safe Unseen Categories**: Encodes unseen categories as all-zeros without crashing.
- **Strict Immutability**: Never mutates the caller's input DataFrame.

---

## AutoML (Baseline Model Training & Evaluation)

`AutoML` provides a leak-free, automated baseline modeling workflow. It automatically determines whether the dataset represents a classification or regression task, trains a standard set of baseline models using `AutoPrep` internally, and evaluates them with standard metrics.

```python
from datacraft import AutoML

automl = AutoML(
    df,
    target="target_column",
    test_size=0.2,
    random_state=42
)
```

### DataCraft Architecture

```text
                 DataCraft
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
  AutoClean       AutoEDA       AutoPrep
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                   AutoML
                     │
              ┌──────┴──────┐
              ↓             ↓
          Classification  Regression
              │             │
              ↓             ↓
          CV + Tuning + Evaluation
                     │
                     ↓
              Save / Load / Predict
```

### AutoML Baseline Models & Tuning Parameters

| Task | Models | Controlled Tuning Hyperparameters |
| :--- | :--- | :--- |
| **Classification** | `LogisticRegression` | `C`, `solver` |
| | `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| | `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| **Regression** | `LinearRegression` | Baseline model (unnecessary tuning omitted) |
| | `DecisionTreeRegressor` | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| | `RandomForestRegressor` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |

### AutoML Evaluation Metrics

| Task | Supported Metrics | Default Primary Metric (`scoring="auto"`) |
| :--- | :--- | :--- |
| **Classification** | `F1`, `accuracy`, `precision`, `recall` | **`F1`** |
| **Regression** | `R²`, `MAE`, `RMSE`, `MSE` | **`R²`** |

### AutoML Methods

| Method | Description | Return Type |
| :--- | :--- | :--- |
| `automl.detect_task()` | Automatically detects `'classification'` or `'regression'` from target properties. | `dict[str, str]` |
| `automl.inspect()` | Inspects sample count, features, target distribution, missing values, and class counts. | `dict[str, Any]` |
| `automl.preview()` | Shows candidate baseline models upfront without training. | `list[str]` |
| `automl.evaluate()` | Performs leak-free K-Fold / StratifiedKFold cross-validation across all baseline models. | `AutoML` |
| `automl.compare()` | Prints and returns a clean comparison table sorted by primary metric, labeling highest-scoring model. | `pd.DataFrame` |
| `automl.tune(n_iter=10)` | Runs controlled `RandomizedSearchCV` inside leak-free pipelines with isolated held-out test split. | `AutoML` |
| `automl.tuning_results()` | Returns summary table with model, best score, best parameters, and iterations. | `pd.DataFrame` |
| `automl.best_models()` | Returns dictionary of tuned pipelines ordered by cross-validation score. | `dict[str, Any]` |
| `automl.evaluate_tuned()` | Evaluates the model with the highest CV score on the held-out test set (or custom test data). | `pd.DataFrame` |
| `automl.predict(X)` | Generates target predictions using the complete preprocessing + model pipeline (DataFrames, Series, arrays). | `np.ndarray` |
| `automl.predict_proba(X)` | Generates predicted class probabilities for classification models. | `np.ndarray` |
| `automl.save_model("model.pkl")` | Persists the entire end-to-end trained pipeline, preprocessing, and metadata using `joblib`. | `str` |
| `AutoML.load_model("model.pkl")` | Loads a saved DataCraft pipeline ready for production prediction without retraining. | `AutoML` |
| `automl.model_info()` | Returns metadata dictionary including task, target, model class name, version, and timestamp. | `dict[str, Any]` |
| `automl.fit()` | Trains baseline models on a single train/test split. | `AutoML` |
| `automl.results()` | Returns evaluation results table (CV fold scores, tuning results, or train/test metrics). | `pd.DataFrame` |
| `automl.report()` | Prints and returns a readable summary of task, status, models, and scores. | `str` |
| `automl.get_models()` | Returns dictionary of fitted scikit-learn model objects (after `fit()`). | `dict[str, Any]` |
| `automl.get_prep()` | Returns the fitted `AutoPrep` instance used internally (after `fit()`). | `AutoPrep` |

---

## Complete End-to-End Workflow Example

The following example demonstrates the complete DataCraft lifecycle on a raw, messy dataset:

$$\text{DataFrame} \longrightarrow \text{AutoClean} \longrightarrow \text{AutoEDA} \longrightarrow \text{AutoPrep} \longrightarrow \text{AutoML} \longrightarrow \text{Tune} \longrightarrow \text{Evaluate} \longrightarrow \text{Save} \longrightarrow \text{Load} \longrightarrow \text{Predict}$$

```python
import pandas as pd
from datacraft import AutoClean, AutoEDA, AutoPrep, AutoML

# Raw messy dataset with missing values, duplicate rows, and inconsistent categories
raw_df = pd.DataFrame({
    "age": [25, 32, None, 51, 62, 23, 38, 45, 56, 29, 34, 41, 25, 32],
    "spend": [120.5, 340.0, 95.0, None, 180.0, 80.0, 210.0, 310.0, 150.0, 190.0, 260.0, 310.0, 120.5, 340.0],
    "region": ["North", "South", "North", "West", "south", "North", "East", "West", "East", "South", "East", "North", "North", "South"],
    "signup_date": pd.date_range("2021-01-01", periods=14, freq="ME"),
    "churn": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
})

# =========================================================================
# Step 1: AutoClean -- Clean missing values, duplicates, and column names
# =========================================================================
cleaner = AutoClean(raw_df, target="churn", clean_col_names=True)
cleaner.inspect()
cleaned_df = cleaner.clean()
cleaner.report()

# =========================================================================
# Step 2: AutoEDA -- Exploratory analysis, correlations, and HTML report
# =========================================================================
eda = AutoEDA(cleaned_df)
eda.inspect()
eda.summary()
eda.correlations(threshold=0.5)
eda.quality()
eda.visualize()
eda.save_report("eda_report.html")

# =========================================================================
# Step 3: AutoPrep -- Leak-free preprocessing inspection & data splitting
# =========================================================================
# Exclude datetime column for standard tabular modeling
modeling_df = cleaned_df.drop(columns=["signup_date"])
prep = AutoPrep(modeling_df, target="churn", test_size=0.2, random_state=42)
prep.inspect()
prep.preview()
X_train_raw, X_test_raw, y_train, y_test = prep.split()

# =========================================================================
# Step 4: AutoML -- Task detection, CV evaluation, controlled tuning & test
# =========================================================================
automl = AutoML(modeling_df, target="churn", cv=3, n_iter=5, scoring="auto", random_state=42)
automl.inspect()
automl.preview()
automl.evaluate()
automl.compare()
automl.tune()
print(automl.tuning_results())
automl.evaluate_tuned()

# =========================================================================
# Step 5: Save & Reload Complete Pipeline for Production Inference
# =========================================================================
# Persists preprocessing (imputation, scaling, encoding) + tuned model
automl.save_model("churn_pipeline.pkl")

# In a separate production session: load model directly without retraining
loaded_automl = AutoML.load_model("churn_pipeline.pkl")
print(loaded_automl.model_info())

# Predict on new unseen raw data with missing values and new categories
new_customer = pd.DataFrame({
    "age": [28.0],
    "spend": [155.0],
    "region": ["South"],
})

predictions = loaded_automl.predict(new_customer)
probabilities = loaded_automl.predict_proba(new_customer)
print("Predicted Churn:", predictions)
print("Predicted Probabilities:", probabilities)
```

---

## Limitations

To maintain simplicity, determinism, and zero external runtime overhead, DataCraft focuses on transparent, robust foundations:
- **Controlled Baselines & Tuning**: Compact hyperparameter spaces optimized for CPU laptops. Does not require heavy distributed clusters or GPUs.
- **No Deep Learning / AutoDL**: DataCraft is dedicated to tabular ML pipelines.
- **No Automated Feature Deletion**: Does not silently drop features without user consent.
- **No Outlier Deletion**: Outliers are flagged mathematically via IQR, never deleted automatically.

---

## Project Structure

```
DataCraft/
├── src/
│   └── datacraft/
│       ├── __init__.py              # Package entry point (exports AutoClean, AutoEDA, AutoPrep, AutoML)
│       ├── cleaner.py               # Core AutoClean implementation
│       ├── eda.py                   # Core AutoEDA implementation
│       ├── prep.py                  # Core AutoPrep implementation (leak-free preprocessing)
│       └── automl.py                # Core AutoML implementation (tuning, CV, persistence & prediction)
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py              # AutoClean unit tests (175+ tests)
│   ├── test_eda.py                  # AutoEDA unit tests (43+ tests)
│   ├── test_prep.py                 # AutoPrep unit tests (26+ tests)
│   ├── test_automl.py               # AutoML unit tests (80+ tests)
│   ├── test_end_to_end.py           # End-to-end integration, API consistency & validation (13+ tests)
│   └── test_real_world.py           # Real-world problem datasets (19+ tests)
├── examples/
│   ├── basic_usage.py               # AutoClean end-to-end usage demonstration
│   ├── eda_usage.py                 # AutoEDA end-to-end usage demonstration
│   ├── prep_usage.py                # AutoPrep ML preprocessing demonstration
│   ├── automl_usage.py              # AutoML CV evaluation, tuning, persistence & prediction
│   ├── real_world_validation.py     # Multi-dataset real-world validation script
│   └── end_to_end.py                # Complete DataCraft v1.0 end-to-end classification & regression workflow
├── pyproject.toml                   # Build metadata & dependency configuration
├── LICENSE                          # MIT License
├── CONTRIBUTING.md                  # Development & contribution guidelines
├── CHANGELOG.md                     # Release version history
├── .gitignore                       # Standard Python ignore rules
└── README.md                        # Documentation
```

---

## Roadmap

- **Phase 13 (Completed)**: **AutoPrep train/test splitting & reproducible pipelines** (leak-free train/test split generator and pipeline persistence).
- **Phase 14 (Completed)**: **AutoML baseline model training & evaluation** (automatic task detection, baseline classification and regression models, metrics table, report).
- **Phase 15 (Completed)**: **Proper cross-validation + model comparison and ranking** (leak-free StratifiedKFold/KFold CV, fold scores, comparison table).
- **Phase 16 (Completed)**: **Controlled hyperparameter tuning with RandomizedSearchCV** (compact search spaces, leak-free pipelines, tuning results, held-out test evaluation, predict).
- **Phase 17 (Completed)**: **Model persistence + production-style prediction** (save/load complete preprocessing + model pipelines without retraining, `predict_proba`, `model_info`).
- **Phase 18 (Completed)**: **AutoML robustness + API cleanup + end-to-end integration** (standardized public API, comprehensive dataset validation, strict reproducibility, zero data leakage tests).
- **Phase 19 (Completed)**: **DataCraft v1.0 Production Release**—verified public APIs across all 4 modules, complete end-to-end examples, clean PyPI build, 100% test pass rate.
- **Next: Real-World Benchmarking & User Feedback**: Validate DataCraft across diverse open-source benchmark datasets, evaluate CPU runtimes against established baselines, and collect developer feedback before expanding scope.
- **Future Considerations**:
  - Expanded imputation strategies (constant fills, time-series forward/backward fills).
  - Optional user-approved category harmonization for casing/whitespace variants.
  - Native Polars DataFrame support.

---

## Running Tests

Run the complete test suite:

```bash
pytest -v --tb=short
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding guidelines, and pull request procedures.

---

## License

MIT (c) 2026 DataCraft contributors.


