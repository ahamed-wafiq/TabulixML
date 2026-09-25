"""
examples/automl_usage.py -- TabulixML AutoML Demonstration

Demonstrates:
  1. Automated task detection (Classification vs Regression).
  2. Inspecting dataset and target characteristics.
  3. Previewing candidate baseline models upfront.
  4. Cross-validation evaluation via evaluate() (StratifiedKFold & KFold) with zero preprocessing leakage.
  5. Model comparison via compare() sorted by primary metric (F1 / R² / custom).
  6. Controlled hyperparameter search via tune() (using RandomizedSearchCV inside leak-free Pipelines).
  7. Tuning results inspection via tuning_results() and best_models().
  8. Generating predictions via predict(X) and held-out test evaluation via evaluate_tuned().
  9. Single train/test split fitting via fit().
  10. Printing comprehensive, human-readable AutoML reports.
"""

import numpy as np
import pandas as pd
from tabulixml import AutoML


def main():
    print("=" * 70)
    print("TabulixML AutoML End-to-End Workflow Demonstration")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Classification with Cross-Validation
    # --------------------------------------------------------------------------
    print("\n>>> Scenario A: Customer Churn Classification (5-Fold Stratified CV)")
    rng = np.random.RandomState(42)
    n = 100

    df_class = pd.DataFrame({
        "age": rng.normal(40, 12, size=n).round(1),
        "monthly_spend": rng.normal(120, 35, size=n).round(2),
        "support_calls": rng.poisson(2, size=n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "churn": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    })

    print(f"Dataset shape: {df_class.shape}")
    print(df_class.head())

    # Initialize AutoML with 5-fold cross-validation and automatic scoring (F1 for classification)
    automl_class = AutoML(
        df_class,
        target="churn",
        cv=5,
        scoring="auto",
        random_state=42,
    )

    # 1. Task detection
    task_info = automl_class.detect_task()
    print(f"\n[1] Task Detected: {task_info}")

    # 2. Inspect dataset
    print("\n[2] Inspecting Dataset:")
    automl_class.inspect()

    # 3. Preview candidate models
    print("\n[3] Previewing Models to Test:")
    automl_class.preview()

    # 4. Cross-validation evaluation (leak-free: ColumnTransformer fitted per training fold)
    print("\n[4] Running 5-Fold Stratified Cross-Validation (evaluate)...")
    automl_class.evaluate()

    # 5. Cross-validation results table
    print("\n[5] Cross-Validation Results Table (Mean F1, Std, Fold 1..5):")
    results_class = automl_class.results()
    print(results_class)

    # 6. Model comparison table (sorted by Mean F1)
    print("\n[6] Model Comparison (compare):")
    automl_class.compare()

    # --------------------------------------------------------------------------
    # 2. Controlled Hyperparameter Tuning
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Scenario B: Controlled Hyperparameter Tuning (tune)")
    print("=" * 70)

    print("\n[1] Running RandomizedSearchCV across compact search spaces...")
    automl_class.tune(n_iter=5)

    print("\n[2] Tuning Results Table:")
    print(automl_class.tuning_results())

    print("\n[3] Tuned Models ordered by Cross-Validation Score:")
    best_models = automl_class.best_models()
    for name, pipeline in best_models.items():
        print(f"  - {name}: {pipeline.named_steps['model'].__class__.__name__}")

    print("\n[4] Evaluating Tuned Model on Held-out Test Set:")
    test_scores = automl_class.evaluate_tuned()

    # 5. Generating predictions for new unseen records
    print("\n[5] Generating Predictions & Probabilities for New Data:")
    new_customers = pd.DataFrame({
        "age": [28.0, 52.0],
        "monthly_spend": [85.50, 160.00],
        "support_calls": [0, 4],
        "contract": ["One year", "Month-to-month"],
    })
    preds = automl_class.predict(new_customers)
    proba = automl_class.predict_proba(new_customers)
    print(f"Predicted churn labels: {preds}")
    print(f"Predicted churn probabilities:\n{proba}")

    # --------------------------------------------------------------------------
    # 3. Model Persistence and Reusable Prediction
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Scenario C: Model Persistence & Reusable Prediction")
    print("=" * 70)

    model_file = "churn_pipeline.pkl"
    print(f"\n[1] Saving complete pipeline to '{model_file}'...")
    saved_path = automl_class.save_model(model_file)
    print(f"Saved successfully to: {saved_path}")

    print("\n[2] Model Metadata (model_info):")
    info = automl_class.model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n[3] Loading model in a fresh session (without retraining)...")
    # Load as class method or on an unconfigured instance
    loaded_automl = AutoML.load_model(model_file)
    print(f"Loaded Task: {loaded_automl.task} | Target: {loaded_automl.target}")

    print("\n[4] Making predictions using loaded pipeline:")
    reloaded_preds = loaded_automl.predict(new_customers)
    reloaded_proba = loaded_automl.predict_proba(new_customers)
    print(f"Reloaded Predictions   : {reloaded_preds}")
    print(f"Reloaded Probabilities :\n{reloaded_proba}")

    # Single-row prediction demonstration
    single_customer = new_customers.iloc[[0]]
    single_pred = loaded_automl.predict(single_customer)
    print(f"Single-row Prediction  : {single_pred}")

    # Clean up demo model file
    from pathlib import Path
    if Path(model_file).exists():
        Path(model_file).unlink()

    # --------------------------------------------------------------------------
    # 4. Regression with Cross-Validation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Scenario D: House Price Regression (5-Fold K-Fold CV & Tuning)")
    print("=" * 70)

    n_reg = 80
    sqft = rng.normal(1800, 400, size=n_reg).round()
    bedrooms = rng.choice([2, 3, 4, 5], size=n_reg)
    neighborhood = rng.choice(["Downtown", "Suburbs", "Rural"], size=n_reg)
    price = (sqft * 180 + bedrooms * 15000 + rng.normal(0, 15000, size=n_reg)).round(2)

    df_reg = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "price": price,
    })

    automl_reg = AutoML(
        df_reg,
        target="price",
        cv=5,
        scoring="auto",  # Defaults to R² for regression
        random_state=42,
    )

    print(f"\n[1] Task Detected: {automl_reg.detect_task()}")
    print("\n[2] Running 5-Fold K-Fold Cross-Validation...")
    automl_reg.evaluate()
    automl_reg.compare()

    print("\n[3] Tuning Regression Models...")
    automl_reg.tune(n_iter=4)
    print(automl_reg.tuning_results())
    automl_reg.evaluate_tuned()

    # --------------------------------------------------------------------------
    # 5. Comprehensive AutoML Report
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> Scenario E: Comprehensive AutoML Report")
    print("=" * 70)

    automl_reg.report()

    print("\nDone! TabulixML AutoML v1.0 executed cleanly with full pipeline persistence.")


if __name__ == "__main__":
    main()

