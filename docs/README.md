# TabulixML Documentation

Welcome to the documentation for **TabulixML** — an automated, lightweight, and deterministic tabular machine learning toolkit built with pure Python, pandas, numpy, and scikit-learn.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Modules](#architecture--modules)
  - [1. AutoClean](#1-autoclean)
  - [2. AutoEDA](#2-autoeda)
  - [3. AutoPrep](#3-autoprep)
  - [4. AutoML](#4-automl)
- [Quick Start](#quick-start)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [API Reference](api_reference.md)

---

## Overview

TabulixML standardizes and streamlines the entire tabular machine learning lifecycle:
1. **Cleaning**: Automated missing-value imputation, duplicate removal, column normalization, and quality audits.
2. **Exploration**: Descriptive statistics, correlation matrices, and automated visualizations.
3. **Preprocessing**: Leakage-free train/test splits and scikit-learn compatible ColumnTransformer pipelines.
4. **Machine Learning**: Automated task detection (classification vs. regression), baseline modeling, cross-validation, hyperparameter tuning, model persistence, and production inference.

---

## Architecture & Modules

```
                    ┌─────────────────┐
                    │  Raw DataFrame  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    AutoClean    │ ─── Non-destructive cleaning & audit
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │     AutoEDA     │               │    AutoPrep     │
   │ Read-only stats │               │ Leak-free split │
   │ & visualization │               │  & transformers │
   └─────────────────┘               └────────┬────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │     AutoML      │
                                     │ CV, Model Eval, │
                                     │  Tuning & Pred  │
                                     └─────────────────┘
```

### 1. AutoClean
Non-destructive data cleaning with full audit logging:
- Strategies for missing value imputation: `mean`, `median`, `mode`, `auto`.
- Deduplication and constant/empty column handling.
- Name normalization to `snake_case`.
- Quality warnings: ID column detection, high-cardinality flags, outlier detection via IQR fences.

### 2. AutoEDA
Exploratory data analysis designed for fast terminal inspections or HTML export:
- Read-only summary statistics.
- Pairwise Pearson correlation matrices with threshold filtering.
- Automated plot generation (histograms, boxplots, heatmaps) without UI blocking.
- Standalone HTML report export via `save_report()`.

### 3. AutoPrep
Leak-free data transformation:
- Automatic detection of numerical, categorical, and datetime columns.
- Reproducible train/test splits.
- Scikit-learn `ColumnTransformer` pipeline generation.
- Transform new unseen test data without data leakage.

### 4. AutoML
Automated tabular model selection, evaluation, hyperparameter tuning, and persistence:
- Automatic task detection (classification vs. regression).
- Baseline models:
  - Classification: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
  - Regression: `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`.
- K-fold and Stratified K-fold cross-validation.
- Hyperparameter tuning using `RandomizedSearchCV`.
- Model persistence (`save_model()`, `load_model()`).
- Direct raw DataFrame inference (`predict()`, `predict_proba()`).

---

## Quick Start

```python
from tabulixml import AutoClean, AutoEDA, AutoPrep, AutoML
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# 1. Clean
cleaner = AutoClean(df)
df_clean = cleaner.clean()

# 2. EDA
eda = AutoEDA(df_clean)
eda.inspect()

# 3. Preprocess
prep = AutoPrep(df_clean, target="target")
prep.prepare()
X_train_trans = prep.transform(prep.X_train)

# 4. AutoML
automl = AutoML(df_clean, target="target", cv=5, random_state=42)
automl.fit()
automl.evaluate()
automl.tune(n_iter=10)
automl.save_model("best_pipeline.joblib")
```

For detailed method signatures, see the [API Reference](api_reference.md).
