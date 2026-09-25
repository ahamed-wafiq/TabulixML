"""
examples/real_world_validation.py -- Realistic Tabular Dataset Validation for DataCraft AutoClean

Demonstrates and validates AutoClean across four real-world dataset scenarios:
  1. Employee dataset: missing values, duplicate rows, categorical columns, dates, numerical outliers
  2. Customer dataset: missing values, inconsistent category casing, high-cardinality, possible ID columns
  3. Sales dataset: numerical columns, dates, duplicate transactions, extreme values
  4. Student dataset: categorical and numerical columns, missing values, constant columns, empty column

For each dataset, runs:
  - inspect()
  - preview()
  - clean()
  - report()
  - history()

Prints a concise validation summary for each dataset:
  Dataset
  Rows before → after
  Columns before → after
  Missing values handled
  Duplicates removed
  Warnings generated
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from datacraft import AutoClean


# ============================================================================
# Dataset Generators
# ============================================================================

def create_employee_dataset() -> pd.DataFrame:
    """
    Employee dataset featuring:
      - Missing values: salary, performance_score, department
      - Duplicate rows: 2 duplicated employee records
      - Categorical columns: department, role
      - Dates: hire_date (datetime)
      - Numerical outliers: extreme salary (2,500,000)
    """
    return pd.DataFrame({
        "emp_name": [
            "Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona",
            "George", "Hannah", "Ian", "Julia", "Alice", "Bob"
        ],
        "department": [
            "Engineering", "Sales", "HR", "Engineering", None, "HR",
            "Sales", "Engineering", "Marketing", "Sales", "Engineering", "Sales"
        ],
        "role": [
            "Senior", "Junior", "Lead", "Junior", "Senior", "Associate",
            "Lead", "Junior", "Senior", "Junior", "Senior", "Junior"
        ],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2021-03-22", "2018-11-05", "2022-07-19", "2019-09-01",
            "2023-02-14", "2017-06-30", "2021-11-11", "2020-08-25", "2022-04-18",
            "2020-01-15", "2021-03-22"
        ]),
        "salary": [
            85000.0, 52000.0, 110000.0, np.nan, 95000.0, 48000.0,
            105000.0, 54000.0, 92000.0, 2500000.0, 85000.0, 52000.0
        ],
        "performance_score": [
            4.2, 3.8, np.nan, 3.5, 4.8, 3.2,
            4.5, 3.9, 4.1, 3.0, 4.2, 3.8
        ]
    })


def create_customer_dataset() -> pd.DataFrame:
    """
    Customer dataset featuring:
      - Missing values: age, annual_spend
      - Inconsistent category casing/whitespace: country variants
      - High-cardinality columns: city (12 unique values)
      - Possible ID columns: customer_id (12 unique values in 12 rows)
    """
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
        "Austin", "Jacksonville"
    ]
    countries = [
        "United States", "united states", "UNITED STATES", "  United States  ",
        "Canada", "canada", "United States", "Canada", "united states",
        "United States", "Canada", "united states"
    ]
    cust_ids = [f"CUST-{i:03d}" for i in range(1, 13)]
    ages = [29.0, np.nan, 42.0, 35.0, 58.0, 24.0, np.nan, 49.0, 33.0, 61.0, 27.0, 38.0]
    annual_spend = [1200.5, 850.0, 2400.0, 450.0, np.nan, 980.0, 3100.0, 1500.0, 720.0, 1950.0, 890.0, 2150.0]

    return pd.DataFrame({
        "customer_id": cust_ids,
        "city": cities,
        "country": countries,
        "age": ages,
        "annual_spend": annual_spend,
    })


def create_sales_dataset() -> pd.DataFrame:
    """
    Sales dataset featuring:
      - Numerical columns: quantity, unit_price, discount_pct, total_amount
      - Dates: transaction_date, ship_date
      - Duplicate transactions: 2 duplicated rows
      - Extreme values / outliers: quantity=5000, total_amount=100000.0
    """
    dates = pd.to_datetime([
        "2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14",
        "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18", "2026-01-19",
        "2026-01-10", "2026-01-11"
    ])
    ship_dates = pd.to_datetime([
        "2026-01-12", "2026-01-13", "2026-01-15", "2026-01-16", "2026-01-18",
        "2026-01-18", "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
        "2026-01-12", "2026-01-13"
    ])
    quantities = [2, 5, 1, 4, 3, 2, 5000, 3, 1, 4, 2, 5]
    unit_prices = [25.0, 10.0, 150.0, 45.0, 80.0, 25.0, 25.0, 60.0, 120.0, 35.0, 25.0, 10.0]
    discount_pct = [0.0, 0.05, 0.10, np.nan, 0.0, 0.05, 0.20, 0.0, 0.15, np.nan, 0.0, 0.05]
    total_amounts = [50.0, 47.5, 135.0, 180.0, 240.0, 47.5, 100000.0, 180.0, 102.0, 140.0, 50.0, 47.5]

    return pd.DataFrame({
        "transaction_date": dates,
        "ship_date": ship_dates,
        "quantity": quantities,
        "unit_price": unit_prices,
        "discount_pct": discount_pct,
        "total_amount": total_amounts,
    })


def create_student_dataset() -> pd.DataFrame:
    """
    Student dataset featuring:
      - Categorical and numerical columns: major, gpa, credits_earned, attendance_pct
      - Missing values: major, gpa, attendance_pct
      - Constant columns: school_code, academic_year (flagged, not dropped)
      - Empty column: empty_notes (dropped)
    """
    return pd.DataFrame({
        "student_name": [
            "Liam", "Emma", "Noah", "Olivia", "William",
            "Ava", "James", "Sophia", "Benjamin", "Isabella"
        ],
        "school_code": ["SCH-101"] * 10,
        "academic_year": [2026] * 10,
        "empty_notes": [np.nan] * 10,
        "major": [
            "Computer Science", "Biology", None, "Economics", "Computer Science",
            "Mathematics", "Physics", None, "Biology", "Economics"
        ],
        "gpa": [3.8, 3.4, 3.9, 2.9, np.nan, 3.7, 3.5, 3.1, np.nan, 3.6],
        "credits_earned": [90, 60, 105, 45, 75, 110, 80, 50, 65, 95],
        "attendance_pct": [95.0, 88.0, 92.0, np.nan, 98.0, 85.0, 90.0, 78.0, 89.0, 93.0],
    })


# ============================================================================
# Validation Pipeline & Summary Extraction
# ============================================================================

def format_warnings(preview_plan: dict) -> list[str]:
    """Extract list of user-facing quality warnings from preview plan."""
    warnings: list[str] = []

    # Constant columns
    if preview_plan.get("constant_cols"):
        c_cols = preview_plan["constant_cols"]
        warnings.append(f"Constant column(s) flagged: {len(c_cols)} ({', '.join(c_cols)})")

    # Inconsistent categories
    if preview_plan.get("inconsistent_categories"):
        inc = preview_plan["inconsistent_categories"]
        warnings.append(f"Inconsistent categorical casing/whitespace: {len(inc)} column(s) ({', '.join(inc.keys())})")

    # Possible ID columns
    if preview_plan.get("id_like_cols"):
        ids = preview_plan["id_like_cols"]
        warnings.append(f"Possible ID column(s) flagged: {len(ids)} ({', '.join(ids)})")

    # High-cardinality categorical columns
    if preview_plan.get("high_cardinality_cols"):
        hc = preview_plan["high_cardinality_cols"]
        warnings.append(f"High-cardinality category column(s): {len(hc)} ({', '.join(hc)})")

    # Redundant numerical features
    if preview_plan.get("redundant_numerical"):
        rn = preview_plan["redundant_numerical"]
        warnings.append(f"Redundant numerical correlation pair(s): {len(rn)}")

    # Outliers
    if preview_plan.get("outliers"):
        outliers = preview_plan["outliers"]
        total_outliers = sum(info["count"] for info in outliers.values())
        out_cols = list(outliers.keys())
        warnings.append(f"Numerical outliers flagged (never deleted): {total_outliers} across {len(out_cols)} column(s) ({', '.join(out_cols)})")

    return warnings


def run_dataset_validation(dataset_name: str, df: pd.DataFrame, clean_col_names: bool = False) -> dict:
    """Run inspect, preview, clean, report, and history on a real-world DataFrame."""
    print("\n" + "=" * 70)
    print(f"  RUNNING VALIDATION: {dataset_name}")
    print("=" * 70)

    # Guarantee immutability check snapshot
    original_snapshot = df.copy()

    cleaner = AutoClean(df, clean_col_names=clean_col_names)

    # 1. inspect()
    info = cleaner.inspect()

    # 2. preview()
    plan = cleaner.preview()

    # 3. clean()
    cleaned_df = cleaner.clean()

    # 4. report()
    cleaner.report()

    # 5. history()
    history_records = cleaner.history()

    # Verification: original DataFrame is unchanged
    pd.testing.assert_frame_equal(df, original_snapshot)

    # Gather metrics
    rows_before, cols_before = original_snapshot.shape
    rows_after, cols_after = cleaned_df.shape

    # Missing values handled count
    missing_handled = sum(
        item["missing_count"]
        for item in cleaner._last_clean_summary.get("imputed", [])
    )

    # Duplicates removed count
    dups_removed = cleaner._last_clean_summary.get("duplicate_rows_removed", 0)

    # Warnings generated
    warnings = format_warnings(plan)

    summary = {
        "dataset": dataset_name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "missing_handled": missing_handled,
        "dups_removed": dups_removed,
        "warnings": warnings,
        "cleaned_df": cleaned_df,
        "history_count": len(history_records),
    }

    return summary


def print_summary(s: dict) -> None:
    """Print the exact required summary format for each dataset."""
    arrow = "→"
    try:
        f"test {arrow}".encode(sys.stdout.encoding or "utf-8")
    except Exception:
        arrow = "->"

    print("\n" + "-" * 70)
    print(f"Dataset                : {s['dataset']}")
    print(f"Rows before {arrow} after    : {s['rows_before']} {arrow} {s['rows_after']}")
    print(f"Columns before {arrow} after : {s['cols_before']} {arrow} {s['cols_after']}")
    print(f"Missing values handled : {s['missing_handled']}")
    print(f"Duplicates removed     : {s['dups_removed']}")
    if s["warnings"]:
        print(f"Warnings generated     : {len(s['warnings'])}")
        for w in s["warnings"]:
            print(f"  - {w}")
    else:
        print("Warnings generated     : 0")
    print("-" * 70)


def main():
    datasets = [
        ("Employee dataset", create_employee_dataset(), False),
        ("Customer dataset", create_customer_dataset(), False),
        ("Sales dataset", create_sales_dataset(), False),
        ("Student dataset", create_student_dataset(), False),
    ]

    summaries = []
    for name, df, clean_names in datasets:
        summary = run_dataset_validation(name, df, clean_col_names=clean_names)
        summaries.append(summary)

    print("\n" + "=" * 70)
    print("  ALL DATASETS VALIDATION SUMMARY")
    print("=" * 70)
    for s in summaries:
        print_summary(s)


if __name__ == "__main__":
    main()
