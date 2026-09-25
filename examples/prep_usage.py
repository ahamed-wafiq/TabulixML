"""
examples/prep_usage.py -- AutoPrep Usage Demonstration for DataCraft

Demonstrates the machine learning preprocessing module (AutoPrep):
  1. inspect() : Inspect feature columns, target, detected types, missing/unique counts
  2. preview() : Review planned imputations, scaling, and encodings upfront
  3. prepare() : Build & fit scikit-learn ColumnTransformer, returning (X, y, pipeline)
  4. transform(): Safely transform new/unseen test data without target leakage
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from datacraft import AutoPrep


def main():
    print("=" * 70)
    print("  DataCraft AutoPrep Demonstration")
    print("=" * 70)

    # 1. Create a realistic training dataset
    train_data = {
        "Age": [25.0, 30.0, np.nan, 45.0, 29.0, 38.0],
        "Salary": [50000.0, 60000.0, 55000.0, 95000.0, 52000.0, 75000.0],
        "Department": ["Engineering", "Sales", "HR", "Sales", np.nan, "Marketing"],
        "Tenure_Years": [2.0, 4.0, 1.0, 12.0, 3.0, 6.0],
        "Joined": pd.to_datetime(["2020-01-15", "2019-06-01", "2021-03-22", "2018-11-05", "2020-08-10", "2019-12-01"]),
        "Promoted": [0, 1, 0, 1, 0, 1],  # Target column
    }
    df_train = pd.DataFrame(train_data)

    print("\nTraining DataFrame (head):")
    print(df_train.head())

    # Initialize AutoPrep with target separation
    prep = AutoPrep(df_train, target="Promoted", scale=True, encode=True)

    # 2. inspect()
    print("\n" + "-" * 70)
    print("1. inspect() - Features, Target & Column Type Breakdown")
    print("-" * 70)
    info = prep.inspect()
    print(f"Target Column      : {info['target_column']}")
    print(f"Feature Columns    : {info['feature_columns']}")
    print(f"Numerical Features : {info['numerical_columns']}")
    print(f"Categorical Features: {info['categorical_columns']}")
    print(f"Datetime Features  : {info['datetime_columns']}")
    print(f"Missing Values     : {info['missing_values']}")

    # 3. preview()
    print("\n" + "-" * 70)
    print("2. preview() - Proposed Preprocessing Pipeline")
    print("-" * 70)
    prep.preview()

    # 4. prepare()
    print("-" * 70)
    print("3. prepare() - Fitting on Train and Transforming Train/Test Splits")
    print("-" * 70)
    X_train, X_test, y_train, y_test = prep.prepare()
    pipeline = prep.get_pipeline()

    print(f"Transformed X_train: shape={X_train.shape}, dtype={X_train.dtype}")
    print(f"Transformed X_test : shape={X_test.shape}, dtype={X_test.dtype}")
    print(f"Target Vector y_train : shape={y_train.shape}, values={list(y_train)}")
    print(f"Fitted Pipeline       :\n{pipeline}")

    # 5. Transforming new / unseen test data
    print("\n" + "-" * 70)
    print("4. Safe Test Data Preprocessing (with unknown category)")
    print("-" * 70)
    test_data = {
        "Age": [32.0],
        "Salary": [68000.0],
        "Department": ["Robotics"],  # Unseen category!
        "Tenure_Years": [5.0],
        "Joined": pd.to_datetime(["2022-04-10"]),
    }
    df_test = pd.DataFrame(test_data)
    print("Unseen Test Sample:")
    print(df_test)

    # Transform test set using AutoPrep's transform()
    X_new = prep.transform(df_test)
    print(f"\nTransformed X_new shape: {X_new.shape}")
    print(f"Transformed X_new values:\n{X_new}")

    # Immutability verification
    print("\nVerification: Original DataFrame shape is strictly unchanged:", df_train.shape)


if __name__ == "__main__":
    main()
