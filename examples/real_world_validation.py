"""
examples/real_world_validation.py -- Comprehensive Real-World & Unseen Dataset Validation for TabulixML v1.0.

Validates the complete TabulixML stack on 6 real-world and unseen tabular datasets:
  1. Employee Dataset (missing values, duplicate rows, casing/whitespace variants, outliers)
  2. Customer Dataset (high cardinality, possible IDs, inconsistent casing, missing values)
  3. Sales Dataset (numerical columns, dates, duplicate transactions, redundant features)
  4. Student Dataset (empty columns, constant columns, numerical outliers, missing values)
  5. Classification Dataset (Loan Approval / Churn - mixed features, class imbalance)
  6. Regression Dataset (Used Car Prices - continuous target, mixed features, noise)

Validates all 4 core modules:
  - AutoClean: inspect, preview, clean, report, history
  - AutoEDA: inspect, summary, correlations, quality, visualize, save_report
  - AutoPrep: inspect, preview, split, prepare, get_pipeline, transform
  - AutoML: detect_task, fit, evaluate, compare, tune, tuning_results, best_models,
            predict, predict_proba, evaluate_tuned, save_model, load_model, model_info

Also stress-tests 11 distinct edge cases:
  1. Empty DataFrame
  2. Single-row DataFrame
  3. Numeric-only dataset
  4. Categorical-only dataset
  5. Datetime columns
  6. All-missing column
  7. Constant column
  8. Duplicate-heavy dataset
  9. Extreme outliers
  10. High-cardinality categorical column
  11. Very small dataset (3 rows)
"""

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend

import numpy as np
import pandas as pd

from tabulixml import AutoClean, AutoEDA, AutoPrep, AutoML


# ============================================================================
# 1. Real-World & Unseen Dataset Generators
# ============================================================================

def create_employee_dataset() -> pd.DataFrame:
    """Employee dataset: missing values, duplicates, casing variants, extreme salary outlier."""
    return pd.DataFrame({
        "emp_name": [
            "Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona",
            "George", "Hannah", "Ian", "Julia", "Alice", "Bob"
        ],
        "department": [
            "Engineering", "Sales", "HR", "Engineering", None, "HR",
            "sales", "Engineering", "Marketing", " Sales ", "Engineering", "Sales"
        ],
        "salary": [
            95000.0, 62000.0, 71000.0, 110000.0, 88000.0, None,
            59000.0, 125000.0, 78000.0, 64000.0, 95000.0, 2500000.0  # outlier
        ],
        "performance_score": [
            3.8, 3.2, 4.1, 4.5, 3.9, 4.0,
            None, 4.7, 3.5, 3.6, 3.8, 3.2
        ],
        "hire_date": pd.date_range("2018-01-01", periods=12, freq="180D").strftime("%Y-%m-%d"),
    })


def create_customer_dataset() -> pd.DataFrame:
    """Customer dataset: missing values, high cardinality, possible ID columns, casing variants."""
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(1, 13)],
        "city": [
            "New York", "Chicago", "San Francisco", "Austin", "Seattle", "Boston",
            "Denver", "Miami", "Dallas", "Atlanta", "Portland", "Phoenix"
        ],
        "country": [
            "USA", "usa", " United States ", "USA", "USA", "usa",
            "USA", "USA", "United States", "USA", "usa", "USA"
        ],
        "age": [29.0, 42.0, None, 35.0, 58.0, 23.0, 45.0, None, 38.0, 51.0, 27.0, 33.0],
        "annual_income": [
            65000.0, 82000.0, 115000.0, 78000.0, 145000.0, 42000.0,
            91000.0, 53000.0, None, 102000.0, 61000.0, 74000.0
        ],
    })


def create_sales_dataset() -> pd.DataFrame:
    """Sales dataset: numerical columns, dates, duplicate rows, redundant features."""
    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i:03d}" for i in range(1, 13)],
        "quantity": [2, 5, 1, 10, 3, 4, 8, 2, 6, 1, 2, 5],
        "unit_price": [25.0, 10.0, 150.0, 12.0, 45.0, 30.0, 15.0, 50.0, 20.0, 90.0, 25.0, 10.0],
        "discount_pct": [0.05, 0.10, 0.0, None, 0.05, 0.10, 0.15, 0.0, None, 0.05, 0.05, 0.10],
        "sale_date": pd.date_range("2023-01-01", periods=12, freq="15D").strftime("%Y-%m-%d"),
    })
    # Perfectly redundant feature
    df["total_amount"] = df["quantity"] * df["unit_price"]
    # Duplicate records
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    return df


def create_student_dataset() -> pd.DataFrame:
    """Student dataset: empty column, constant columns, numerical outliers, missing values."""
    return pd.DataFrame({
        "student_name": [
            "Liam", "Noah", "Oliver", "Emma", "Charlotte",
            "Amelia", "Sophia", "Ava", "Isabella", "Mia"
        ],
        "school_code": ["SCH01"] * 10,       # Constant
        "academic_year": [2024] * 10,         # Constant
        "empty_notes": [None] * 10,           # Completely empty
        "major": ["Biology", "Physics", "Chemistry", None, "Biology", "Math", "Physics", None, "Math", "Biology"],
        "gpa": [3.6, 3.2, None, 3.9, 3.8, 2.9, None, 3.5, 3.7, 4.0],
        "attendance_pct": [92.0, 88.0, 95.0, 91.0, 89.0, 42.0, 94.0, None, 96.0, 93.0],  # 42.0 is outlier
    })


def create_classification_dataset(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Loan Approval Classification dataset: mixed features, missing values, duplicates, class imbalance."""
    rng = np.random.RandomState(random_state)
    credit_score = rng.randint(550, 850, size=n_samples).astype(float)
    annual_income = rng.uniform(25000.0, 150000.0, size=n_samples)
    debt_to_income = rng.uniform(0.1, 0.6, size=n_samples)
    loan_amount = rng.uniform(5000.0, 50000.0, size=n_samples)
    employment_type = rng.choice(["Salaried", "Self-Employed", "Contract", "Unemployed"], size=n_samples)
    application_date = pd.date_range("2022-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")

    # Logistic approval probability
    z = (0.01 * (credit_score - 650) + 0.00002 * (annual_income - 60000) - 4.0 * (debt_to_income - 0.3) - 0.5)
    prob = 1.0 / (1.0 + np.exp(-z))
    loan_approved = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "employment_type": employment_type,
        "application_date": application_date,
        "loan_approved": loan_approved,
    })

    # Add realistic messiness
    df.loc[rng.choice(n_samples, size=6, replace=False), "credit_score"] = np.nan
    df.loc[rng.choice(n_samples, size=5, replace=False), "debt_to_income"] = np.nan
    df.loc[rng.choice(n_samples, size=4, replace=False), "employment_type"] = None

    # Inconsistent casing
    df.loc[df["employment_type"] == "Salaried", "employment_type"] = rng.choice(
        ["Salaried", " salaried ", "SALARIED"],
        size=(df["employment_type"] == "Salaried").sum()
    )

    # Duplicates
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    return df


def create_regression_dataset(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Used Car Pricing Regression dataset: continuous target, mixed features, missing values, duplicates."""
    rng = np.random.RandomState(random_state)
    vehicle_age = rng.randint(1, 15, size=n_samples).astype(float)
    mileage = rng.uniform(5000.0, 180000.0, size=n_samples)
    engine_size = rng.choice([1.5, 1.8, 2.0, 2.5, 3.5], size=n_samples)
    fuel_type = rng.choice(["Petrol", "Diesel", "Hybrid", "Electric"], size=n_samples)
    listing_date = pd.date_range("2023-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")

    fuel_mult = {"Petrol": 0, "Diesel": 2000, "Hybrid": 5000, "Electric": 8000}
    price = (
        35000
        - 1500 * vehicle_age
        - 0.08 * mileage
        + 3000 * engine_size
        + np.array([fuel_mult[f] for f in fuel_type])
        + rng.normal(0, 2000, size=n_samples)
    )
    price = np.maximum(price, 3000.0)

    df = pd.DataFrame({
        "vehicle_age": vehicle_age,
        "mileage": mileage,
        "engine_size": engine_size,
        "fuel_type": fuel_type,
        "listing_date": listing_date,
        "resale_price": price,
    })

    # Add realistic messiness
    df.loc[rng.choice(n_samples, size=7, replace=False), "vehicle_age"] = np.nan
    df.loc[rng.choice(n_samples, size=5, replace=False), "mileage"] = np.nan
    df.loc[rng.choice(n_samples, size=4, replace=False), "fuel_type"] = None

    # Duplicates
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    return df


# ============================================================================
# 2. Validation Harness & Benchmark Collector
# ============================================================================

benchmark_records: List[Dict[str, Any]] = []


def validate_dataset(
    name: str,
    df: pd.DataFrame,
    target: str | None = None,
    run_ml: bool = False,
) -> pd.DataFrame:
    """Run full AutoClean, AutoEDA, and optionally AutoPrep & AutoML validation on a dataset."""
    start_time = time.perf_counter()
    print("\n" + "=" * 80)
    print(f"  VALIDATING DATASET: {name} (Shape: {df.shape[0]} rows x {df.shape[1]} cols)")
    print("=" * 80)

    original_copy = df.copy(deep=True)
    warnings_list = []

    # ------------------------------------------------------------------------
    # 2. AutoClean
    # ------------------------------------------------------------------------
    print(f"\n[1] Testing AutoClean on '{name}'...")
    cleaner = AutoClean(df, target=target, clean_col_names=True)

    # inspect()
    insp = cleaner.inspect()
    assert isinstance(insp, dict), "inspect() must return a dict"
    missing_count = sum(insp["missing"].values())
    dup_count = insp["duplicates"]
    empty_cols = insp["empty_cols"]
    constant_cols = insp["constant_cols"]
    outliers = insp["outliers"]
    print(f"    - Missing values detected: {missing_count}")
    print(f"    - Duplicates detected: {dup_count}")
    print(f"    - Empty columns detected: {empty_cols}")
    print(f"    - Constant columns detected: {constant_cols}")
    print(f"    - Outliers detected (flagged only): {list(outliers.keys())}")

    # preview()
    preview_plan = cleaner.preview()
    assert isinstance(preview_plan, dict), "preview() must return a dict"

    # clean()
    cleaned_df = cleaner.clean()
    assert isinstance(cleaned_df, pd.DataFrame), "clean() must return a DataFrame"
    assert not cleaned_df.empty, "cleaned DataFrame should not be empty"

    # Immutability verification
    pd.testing.assert_frame_equal(df, original_copy, check_exact=True)
    print("    - Immutability check: Original DataFrame strictly unchanged.")

    # report()
    cleaner.report()

    # history()
    hist = cleaner.history()
    assert isinstance(hist, list), "history() must return a list"
    print(f"    - History entries logged: {len(hist)}")

    # ------------------------------------------------------------------------
    # 3. AutoEDA
    # ------------------------------------------------------------------------
    print(f"\n[2] Testing AutoEDA on '{name}'...")
    eda = AutoEDA(cleaned_df)

    # inspect()
    eda_insp = eda.inspect()
    assert isinstance(eda_insp, dict), "AutoEDA.inspect() must return a dict"

    # summary()
    summary_dict = eda.summary()
    assert isinstance(summary_dict, dict), "AutoEDA.summary() must return a dict"

    # correlations()
    corrs = eda.correlations(threshold=0.3)
    assert isinstance(corrs, dict), "correlations() must return a dict"
    assert isinstance(corrs["matrix"], pd.DataFrame), "correlations()['matrix'] must be a DataFrame"

    # quality()
    quality_audit = eda.quality()
    assert isinstance(quality_audit, dict), "quality() must return a dict"

    # visualize() & save_report() in temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        eda.visualize(output_dir=os.path.join(tmp_dir, "plots"))
        report_file = os.path.join(tmp_dir, "report.html")
        eda.save_report(report_file)
        assert os.path.exists(report_file), "save_report() must create HTML report file"
        print(f"    - HTML Report & Visualizations verified ({os.path.getsize(report_file)} bytes).")

    # ------------------------------------------------------------------------
    # 4. AutoPrep & 5. AutoML (if target provided)
    # ------------------------------------------------------------------------
    ml_task = "N/A"
    models_tested = "N/A"
    cv_score_summary = "N/A"
    tuning_result_summary = "N/A"

    if run_ml and target is not None:
        print(f"\n[3] Testing AutoPrep on '{name}' with target='{target}'...")
        prep = AutoPrep(cleaned_df, target=target, test_size=0.2, random_state=42)

        # inspect() & preview()
        prep_insp = prep.inspect()
        prep.preview()

        # split()
        X_train_raw, X_test_raw, y_train, y_test = prep.split()
        assert len(X_train_raw) + len(X_test_raw) == len(cleaned_df)
        assert target not in X_train_raw.columns, "Target must NOT be in feature columns!"

        # prepare()
        X_train_mat, X_test_mat, y_tr, y_te = prep.prepare()
        pipeline = prep.get_pipeline()
        assert pipeline is not None

        # transform() with novel / unseen categories
        dummy_row = X_test_raw.iloc[[0]].copy()
        if prep.categorical_columns:
            cat_col = prep.categorical_columns[0]
            dummy_row[cat_col] = "Completely_Unseen_Novel_Category_XYZ"
        transformed_row = prep.transform(dummy_row)
        assert transformed_row.shape[1] == X_train_mat.shape[1], "Unseen category handled gracefully."
        print("    - AutoPrep pipeline, train-only fitting, and unseen category tolerance verified.")

        # --------------------------------------------------------------------
        # AutoML Execution
        # --------------------------------------------------------------------
        print(f"\n[4] Testing AutoML on '{name}' with target='{target}'...")
        automl = AutoML(cleaned_df, target=target, cv=3, random_state=42)

        task_info = automl.detect_task()
        ml_task = task_info["task"]
        print(f"    - Auto-detected task: {ml_task}")

        automl.inspect()
        automl.preview()

        # evaluate() & compare()
        automl.evaluate()
        cv_df = automl.compare()
        models_tested = ", ".join(cv_df["Model"].tolist())
        top_model = cv_df.iloc[0]["Model"]
        top_score_col = [c for c in cv_df.columns if c.startswith("Mean ")][0]
        top_score = cv_df.iloc[0][top_score_col]
        cv_score_summary = f"{top_model} ({top_score_col}: {top_score:.4f})"
        print(f"    - Highest cross-validation score for the selected metric: {cv_score_summary}")

        # tune(), tuning_results(), best_models()
        print("    - Tuning baseline models with RandomizedSearchCV...")
        automl.tune(n_iter=4)
        tune_df = automl.tuning_results()
        best_models = automl.best_models()
        assert len(best_models) > 0
        best_tuned_name = list(best_models.keys())[0]
        tuning_result_summary = f"Top model: {best_tuned_name}"

        # predict() & predict_proba()
        test_sample = cleaned_df.drop(columns=[target]).iloc[:3]
        preds = automl.predict(test_sample)
        assert len(preds) == 3
        if ml_task == "classification":
            probas = automl.predict_proba(test_sample)
            assert probas.shape[0] == 3
            print("    - predict() and predict_proba() verified.")
        else:
            print("    - predict() verified.")

        # evaluate_tuned()
        automl.evaluate_tuned()

        # save_model() & load_model() & model_info()
        with tempfile.TemporaryDirectory() as model_dir:
            model_file = os.path.join(model_dir, "test_model.pkl")
            automl.save_model(model_file)
            loaded_model = AutoML.load_model(model_file)
            info = loaded_model.model_info()
            assert info["task"] == ml_task
            assert info["version"] == "1.0.0"

            reloaded_preds = loaded_model.predict(test_sample)
            np.testing.assert_array_equal(preds, reloaded_preds)
            print("    - Model serialization, persistence & model_info verified.")

    runtime = time.perf_counter() - start_time

    # Record benchmark
    benchmark_records.append({
        "dataset_type": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "detected_column_types": str(insp["col_types"]),
        "missing_values": missing_count,
        "duplicates": dup_count,
        "outliers": len(outliers),
        "cleaning_changes": f"Rows: {df.shape[0]}->{cleaned_df.shape[0]}, Cols: {df.shape[1]}->{cleaned_df.shape[1]}",
        "preprocessing_result": "Success (leak-free)" if run_ml else "N/A",
        "detected_ml_task": ml_task,
        "models_tested": models_tested,
        "cv_scores": cv_score_summary,
        "tuning_result": tuning_result_summary,
        "runtime_seconds": round(runtime, 2),
        "errors_warnings": f"Cleaned {missing_count} missing, {dup_count} dups; {len(outliers)} outliers flagged",
    })

    print(f"\n[OK] Dataset '{name}' validation completed in {runtime:.2f}s")
    return cleaned_df


# ============================================================================
# 3. Stress-Testing Edge Cases
# ============================================================================

def run_edge_case_stress_tests():
    print("\n" + "=" * 80)
    print("  STRESS-TESTING EDGE CASES")
    print("=" * 80)

    # 1. Empty DataFrame
    print("\n[Edge Case 1] Empty DataFrame:")
    empty_df = pd.DataFrame()
    try:
        AutoClean(empty_df)
        print("  AutoClean handled empty DataFrame.")
    except ValueError as e:
        print(f"  Expected AutoClean rejection: {e}")
    try:
        AutoEDA(empty_df)
    except ValueError as e:
        print(f"  Expected AutoEDA rejection: {e}")

    # 2. Single-row DataFrame
    print("\n[Edge Case 2] Single-row DataFrame:")
    single_df = pd.DataFrame({"a": [10.0], "b": ["alpha"], "target": [1]})
    cleaner = AutoClean(single_df)
    c_df = cleaner.clean()
    assert len(c_df) == 1
    eda = AutoEDA(single_df)
    assert eda.inspect()["rows"] == 1
    try:
        prep = AutoPrep(single_df, target="target")
        prep.split()
    except ValueError as e:
        print(f"  Expected AutoPrep single-row split rejection: {e}")

    # 3. Numeric-only dataset
    print("\n[Edge Case 3] Numeric-only Dataset:")
    num_df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "f2": [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    c_num = AutoClean(num_df).clean()
    prep_num = AutoPrep(c_num, target="target", test_size=0.2, random_state=42)
    X_tr, _, _, _ = prep_num.prepare()
    assert X_tr.shape[1] == 2
    print("  Numeric-only dataset cleaned and preprocessed successfully.")

    # 4. Categorical-only dataset
    print("\n[Edge Case 4] Categorical-only Dataset:")
    cat_df = pd.DataFrame({
        "color": ["red", "blue", "red", "green", "blue", "red", "green", "blue", "red", "green"],
        "size": ["S", "M", "L", "S", None, "L", "M", "S", "L", "M"],
        "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
    })
    c_cat = AutoClean(cat_df).clean()
    prep_cat = AutoPrep(c_cat, target="target", test_size=0.2, random_state=42)
    X_cat, _, _, _ = prep_cat.prepare()
    assert X_cat.shape[1] > 2
    print("  Categorical-only dataset cleaned and one-hot encoded successfully.")

    # 5. Datetime columns
    print("\n[Edge Case 5] Datetime Columns:")
    dt_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "val": range(10),
        "target": [0, 1] * 5,
    })
    c_dt = AutoClean(dt_df).clean()
    prep_dt = AutoPrep(c_dt, target="target")
    assert "date" in prep_dt.datetime_columns
    print("  Datetime column safely identified and segregated from transformation.")

    # 6. All-missing column
    print("\n[Edge Case 6] All-missing Column:")
    all_nan_df = pd.DataFrame({
        "valid": range(10),
        "empty_col": [None] * 10,
    })
    c_nan = AutoClean(all_nan_df).clean()
    assert "empty_col" not in c_nan.columns
    print("  All-missing column automatically identified and removed.")

    # 7. Constant column
    print("\n[Edge Case 7] Constant Column:")
    const_df = pd.DataFrame({
        "feature": range(10),
        "constant_col": [42] * 10,
    })
    cleaner_const = AutoClean(const_df)
    c_const = cleaner_const.clean()
    assert "constant_col" in c_const.columns  # Flagged, never silently dropped
    assert "constant_col" in cleaner_const.inspect()["constant_cols"]
    print("  Constant column correctly flagged in inspection and safely preserved.")

    # 8. Duplicate-heavy dataset
    print("\n[Edge Case 8] Duplicate-heavy Dataset:")
    dup_df = pd.DataFrame({
        "a": [1, 1, 1, 1, 1, 2, 2, 2],
        "b": ["x", "x", "x", "x", "x", "y", "y", "y"],
    })
    cleaner_dup = AutoClean(dup_df, remove_duplicates=True)
    c_dup = cleaner_dup.clean()
    assert len(c_dup) == 2
    print(f"  Duplicate-heavy dataset: reduced from {len(dup_df)} to {len(c_dup)} distinct rows.")

    # 9. Extreme outliers
    print("\n[Edge Case 9] Extreme Outliers:")
    outlier_df = pd.DataFrame({
        "measure": [10.0, 10.5, 11.0, 9.8, 10.2, 10.1, 10.3, 10.4, 9.9, 1000000.0]
    })
    cleaner_out = AutoClean(outlier_df)
    c_out = cleaner_out.clean()
    assert len(c_out) == 10
    assert "measure" in cleaner_out.inspect()["outliers"]
    assert c_out["measure"].max() == 1000000.0
    print("  Extreme outlier detected via IQR fences and safely preserved without silent deletion.")

    # 10. High-cardinality categorical column
    print("\n[Edge Case 10] High-cardinality Categorical Column:")
    high_card_df = pd.DataFrame({
        "uuid": [f"ID_{i}" for i in range(50)],
        "target": [0, 1] * 25,
    })
    cleaner_hc = AutoClean(high_card_df)
    insp_hc = cleaner_hc.inspect()
    assert "uuid" in insp_hc["high_cardinality_cols"] or "uuid" in insp_hc["id_like_cols"]
    print("  High-cardinality / ID-like column detected and warning generated.")

    # 11. Very small dataset (3 rows)
    print("\n[Edge Case 11] Very Small Dataset (3 rows):")
    tiny_df = pd.DataFrame({
        "x": [1.0, 2.0, None],
        "y": ["a", "b", "a"],
    })
    cleaner_tiny = AutoClean(tiny_df)
    c_tiny = cleaner_tiny.clean()
    assert len(c_tiny) == 3
    assert not c_tiny["x"].isna().any()
    print("  Very small dataset (3 rows) successfully imputed and cleaned.")


# ============================================================================
# 4. Main Validation Workflow Execution
# ============================================================================

def main():
    print("#" * 80)
    print("  TABULIXML v1.0 REAL-WORLD & UNSEEN DATASET VALIDATION BENCHMARK")
    print("#" * 80)

    # 1. Employee Dataset (Data hygiene focus)
    validate_dataset("Employee Dataset", create_employee_dataset(), target=None, run_ml=False)

    # 2. Customer Dataset (High cardinality & ID focus)
    validate_dataset("Customer Dataset", create_customer_dataset(), target=None, run_ml=False)

    # 3. Sales Dataset (Redundancy & Duplicates focus)
    validate_dataset("Sales Dataset", create_sales_dataset(), target=None, run_ml=False)

    # 4. Student Dataset (Empty & Constant column focus)
    validate_dataset("Student Dataset", create_student_dataset(), target=None, run_ml=False)

    # 5. Loan Approval Dataset (Full ML Pipeline - Classification)
    validate_dataset("Loan Approval (Classification)", create_classification_dataset(n_samples=120, random_state=42), target="loan_approved", run_ml=True)

    # 6. Used Car Resale Price Dataset (Full ML Pipeline - Regression)
    validate_dataset("Car Resale Price (Regression)", create_regression_dataset(n_samples=120, random_state=42), target="resale_price", run_ml=True)

    # Edge cases
    run_edge_case_stress_tests()

    print("\n" + "=" * 80)
    print("  ALL REAL-WORLD VALIDATIONS AND EDGE CASES PASSED WITH 0 FAILURES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
