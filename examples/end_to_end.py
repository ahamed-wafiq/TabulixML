"""
examples/end_to_end.py -- Complete End-to-End DataCraft v1.0 Demonstration.

Demonstrates the entire DataCraft pipeline from raw messy data to production inference:
  DataFrame -> AutoClean -> AutoEDA -> AutoPrep -> AutoML -> CV -> Tune -> Evaluate -> Save -> Load -> Predict

Covers both:
  1. Classification (Customer Churn)
  2. Regression (House Prices)

Fast and CPU-friendly execution using small synthetic datasets.
"""

import os
import tempfile
import numpy as np
import pandas as pd

from datacraft import AutoClean, AutoEDA, AutoPrep, AutoML


# =====================================================================
# 1. Dataset Generators (Classification & Regression)
# =====================================================================

def make_churn_dataset(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic, messy classification dataset for churn prediction."""
    rng = np.random.RandomState(random_state)

    age = rng.randint(18, 70, size=n_samples).astype(float)
    monthly_charges = rng.uniform(20.0, 120.0, size=n_samples)
    tenure_months = rng.randint(1, 60, size=n_samples).astype(float)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples)
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n_samples)
    signup_date = pd.date_range("2021-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")

    # Target calculation with realistic noise
    churn_prob = 1.0 / (1.0 + np.exp(-(0.03 * (70 - age) + 0.02 * monthly_charges - 0.05 * tenure_months - 0.5)))
    churn = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "Customer Age": age,
        "Monthly_Charges": monthly_charges,
        "Tenure Months": tenure_months,
        "Contract_Type": contract_type,
        "Payment Method": payment_method,
        "Signup Date": signup_date,
        "churn": churn,
    })

    # Inject realistic messiness: missing values, whitespace, duplicates
    df.loc[rng.choice(n_samples, size=8, replace=False), "Customer Age"] = np.nan
    df.loc[rng.choice(n_samples, size=6, replace=False), "Monthly_Charges"] = np.nan
    df.loc[rng.choice(n_samples, size=5, replace=False), "Contract_Type"] = None

    # Inconsistent whitespace/casing
    df.loc[df["Contract_Type"] == "Month-to-month", "Contract_Type"] = rng.choice(
        ["Month-to-month", "month-to-month ", " Month-to-Month"],
        size=(df["Contract_Type"] == "Month-to-month").sum(),
    )

    # Duplicate rows
    duplicates = df.iloc[:4].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def make_housing_dataset(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic, messy regression dataset for house price prediction."""
    rng = np.random.RandomState(random_state)

    sqft = rng.randint(600, 3500, size=n_samples).astype(float)
    bedrooms = rng.randint(1, 5, size=n_samples).astype(float)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n_samples)
    neighborhood = rng.choice(["Downtown", "Suburbs", "Rural"], size=n_samples)
    construction_date = pd.date_range("2010-01-01", periods=n_samples, freq="W").strftime("%Y-%m-%d")

    # Continuous target with realistic noise
    neighborhood_premium = {"Downtown": 80000, "Suburbs": 40000, "Rural": 0}
    price = (
        100000
        + 120 * sqft
        + 15000 * bedrooms
        + 20000 * bathrooms
        + np.array([neighborhood_premium[n] for n in neighborhood])
        + rng.normal(0, 15000, size=n_samples)
    )

    df = pd.DataFrame({
        "Square Feet": sqft,
        "Bedrooms Count": bedrooms,
        "Bathrooms": bathrooms,
        "Neighborhood_Zone": neighborhood,
        "Build Date": construction_date,
        "sale_price": price,
    })

    # Inject messiness
    df.loc[rng.choice(n_samples, size=7, replace=False), "Square Feet"] = np.nan
    df.loc[rng.choice(n_samples, size=5, replace=False), "Neighborhood_Zone"] = None

    # Duplicate rows
    duplicates = df.iloc[:3].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


# =====================================================================
# 2. Complete End-to-End Workflow Runner
# =====================================================================

def run_classification_pipeline():
    print("\n" + "=" * 70)
    print("  DATACRAFT v1.0 -- END-TO-END CLASSIFICATION PIPELINE")
    print("=" * 70)

    # Step 1: Create Raw Messy Data
    raw_df = make_churn_dataset(n_samples=120, random_state=42)
    print(f"\n[1] Raw Dataset Created: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    # Step 2: AutoClean
    print("\n[2] AutoClean: Inspecting, Previewing & Cleaning Data...")
    cleaner = AutoClean(raw_df, target="churn", clean_col_names=True)
    inspect_data = cleaner.inspect()
    total_missing = sum(inspect_data["missing"].values())
    print(f"    - Missing values detected: {total_missing}")
    print(f"    - Duplicates detected: {inspect_data['duplicates']}")

    cleaned_df = cleaner.clean()
    print(f"    - Cleaned shape: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns")
    print(f"    - Cleaned column names: {list(cleaned_df.columns)}")
    history_entries = cleaner.history()
    print(f"    - Audit history operations: {len(history_entries)}")

    # Step 3: AutoEDA
    print("\n[3] AutoEDA: Diagnostic Summaries & HTML Report...")
    eda = AutoEDA(cleaned_df)
    summary_dict = eda.summary()
    print(f"    - Numerical summary columns: {list(summary_dict['numerical'].index)}")
    corrs = eda.correlations(threshold=0.3)
    print(f"    - High correlation pairs (|r| >= 0.3): {len(corrs)}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "eda_report.html")
        eda.save_report(report_path)
        print(f"    - HTML EDA Report exported: {os.path.basename(report_path)} ({os.path.getsize(report_path)} bytes)")

    # Step 4: AutoPrep
    print("\n[4] AutoPrep: Leak-Free Splitting & Preprocessing Pipeline...")
    prep = AutoPrep(cleaned_df, target="churn", test_size=0.2, random_state=42)
    prep_info = prep.inspect()
    print(f"    - Detected numerical: {prep_info['numerical_columns']}")
    print(f"    - Detected categorical: {prep_info['categorical_columns']}")
    print(f"    - Detected datetime: {prep_info['datetime_columns']}")

    X_train, X_test, y_train, y_test = prep.prepare()
    print(f"    - Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")
    print(f"    - Transformed feature matrix width: {X_train.shape[1]} features")

    # Step 5: AutoML (Cross-Validation, Tuning, Persistence, Prediction)
    print("\n[5] AutoML: Cross-Validation, Tuning & Model Selection...")
    automl = AutoML(cleaned_df, target="churn", cv=3, random_state=42)
    task_info = automl.detect_task()
    print(f"    - Auto-detected task: {task_info['task']} (target: '{task_info['target']}')")

    # Cross-validation baseline comparison
    automl.evaluate()
    cv_results = automl.compare()
    print("\n    Cross-Validation Leaderboard:")
    print(cv_results)

    # Hyperparameter tuning
    print("\n    Tuning Best Models with RandomizedSearchCV...")
    automl.tune(n_iter=5)
    tuning_df = automl.tuning_results()
    print("\n    Tuned Models Leaderboard:")
    print(tuning_df)
    best_pipelines = automl.best_models()
    best_model_name = list(best_pipelines.keys())[0]
    print(f"    - Best tuned model: {best_model_name}")

    # Model info & persistence
    with tempfile.TemporaryDirectory() as model_dir:
        model_path = os.path.join(model_dir, "churn_model.pkl")
        automl.save_model(model_path)
        print(f"    - Saved pipeline artifact to: {os.path.basename(model_path)}")

        # Step 6: Load & Production Inference
        print("\n[6] Production Inference with Loaded Model:")
        loaded_automl = AutoML.load_model(model_path)
        info = loaded_automl.model_info()
        print(f"    - Loaded model: {info['model']} (DataCraft v{info['version']})")

        # Test inference on raw unseen data (including missing values and novel category)
        sample_raw = pd.DataFrame([{
            "customer_age": 45.0,
            "monthly_charges": 85.50,
            "tenure_months": 12.0,
            "contract_type": "Two year",
            "payment_method": "New Unseen Wallet",
            "signup_date": "2023-05-15",
        }])

        prediction = loaded_automl.predict(sample_raw)
        probabilities = loaded_automl.predict_proba(sample_raw)
        print(f"    - Input sample: Age=45, Charges=$85.50, Tenure=12mo, Contract='Two year'")
        print(f"    - Predicted class: {prediction[0]} (Churn={bool(prediction[0])})")
        print(f"    - Predicted probabilities: [No Churn: {probabilities[0][0]:.3f}, Churn: {probabilities[0][1]:.3f}]")


def run_regression_pipeline():
    print("\n" + "=" * 70)
    print("  DATACRAFT v1.0 -- END-TO-END REGRESSION PIPELINE")
    print("=" * 70)

    # Step 1: Create Messy Regression Dataset
    raw_df = make_housing_dataset(n_samples=100, random_state=42)
    print(f"\n[1] Raw Dataset Created: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    # Step 2: AutoClean
    cleaner = AutoClean(raw_df, target="sale_price", clean_col_names=True)
    cleaned_df = cleaner.clean()
    print(f"[2] AutoClean: cleaned shape: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns")

    # Step 3: AutoML (Evaluate & Tune)
    print("\n[3] AutoML: Auto-detect Task, Evaluate & Tune Regression Models...")
    automl = AutoML(cleaned_df, target="sale_price", cv=3, random_state=42)
    task_info = automl.detect_task()
    print(f"    - Auto-detected task: {task_info['task']} (target: '{task_info['target']}')")

    automl.evaluate()
    cv_results = automl.compare()
    print("\n    Cross-Validation Leaderboard (Regression):")
    print(cv_results)

    print("\n    Tuning Regression Models...")
    automl.tune(n_iter=5)
    tuning_df = automl.tuning_results()
    print("\n    Tuned Regression Leaderboard:")
    print(tuning_df)
    best_pipelines = automl.best_models()
    best_model_name = list(best_pipelines.keys())[0]
    print(f"    - Top Model: {best_model_name}")

    # Step 4: Save & Predict
    with tempfile.TemporaryDirectory() as model_dir:
        model_path = os.path.join(model_dir, "housing_model.pkl")
        automl.save_model(model_path)

        loaded_automl = AutoML.load_model(model_path)
        sample_house = pd.DataFrame([{
            "square_feet": 2200.0,
            "bedrooms_count": 3.0,
            "bathrooms": 2.0,
            "neighborhood_zone": "Downtown",
            "build_date": "2015-06-01",
        }])

        predicted_price = loaded_automl.predict(sample_house)
        print(f"\n[4] Production Prediction on House (2200 sqft, 3 bed, 2 bath, Downtown):")
        print(f"    - Estimated Value: ${predicted_price[0]:,.2f}")


# =====================================================================
# Main Entry Point
# =====================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  RUNNING DATACRAFT v1.0 COMPLETE END-TO-END DEMONSTRATIONS")
    print("#" * 70)

    run_classification_pipeline()
    run_regression_pipeline()

    print("\n" + "=" * 70)
    print("  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")
