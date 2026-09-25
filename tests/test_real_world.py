"""
tests/test_real_world.py -- Realistic Tabular Dataset Tests for TabulixML AutoClean

Validates AutoClean on four real-world datasets:
  1. Employee dataset: missing values, duplicate rows, categorical columns, dates, numerical outliers
  2. Customer dataset: missing values, inconsistent category casing, high-cardinality, possible ID columns
  3. Sales dataset: numerical columns, dates, duplicate transactions, extreme values
  4. Student dataset: categorical & numerical columns, missing values, constant columns, empty column
"""

import numpy as np
import pandas as pd
import pytest
from tabulixml import AutoClean


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def employee_df() -> pd.DataFrame:
    """Employee dataset fixture."""
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


@pytest.fixture()
def customer_df() -> pd.DataFrame:
    """Customer dataset fixture."""
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


@pytest.fixture()
def sales_df() -> pd.DataFrame:
    """Sales dataset fixture."""
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


@pytest.fixture()
def student_df() -> pd.DataFrame:
    """Student dataset fixture."""
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
# 1. Employee Dataset Test Suite
# ============================================================================

class TestRealWorldEmployeeDataset:
    def test_workflow_runs_and_preserves_input(self, employee_df, capsys):
        snap = employee_df.copy()
        cleaner = AutoClean(employee_df)

        info = cleaner.inspect()
        assert info["duplicates"] == 2
        assert "salary" in info["outliers"]

        plan = cleaner.preview()
        assert plan["duplicate_rows"] == 2

        cleaned = cleaner.clean()
        assert isinstance(cleaned, pd.DataFrame)

        cleaner.report()
        out = capsys.readouterr().out
        assert "TabulixML -- AutoClean Report" in out

        hist = cleaner.history()
        assert len(hist) > 0

        # Verification: input is never mutated
        pd.testing.assert_frame_equal(employee_df, snap)

    def test_employee_duplicates_removed(self, employee_df):
        cleaner = AutoClean(employee_df)
        cleaned = cleaner.clean()
        # Original has 12 rows with 2 exact duplicates -> cleaned has 10 rows
        assert len(cleaned) == 10
        assert cleaned.duplicated().sum() == 0

    def test_employee_missing_values_handled(self, employee_df):
        cleaner = AutoClean(employee_df)
        cleaned = cleaner.clean()
        # Zero missing values across all columns
        assert cleaned.isnull().sum().sum() == 0
        # Categorical column 'department' imputed with mode
        assert "Engineering" in cleaned["department"].values or "Sales" in cleaned["department"].values

    def test_employee_outliers_flagged_not_deleted(self, employee_df):
        cleaner = AutoClean(employee_df)
        plan = cleaner.preview()
        assert "salary" in plan["outliers"]
        assert plan["outliers"]["salary"]["count"] >= 1

        cleaned = cleaner.clean()
        # Outlier row (salary = 2,500,000) must still exist in cleaned data
        assert (cleaned["salary"] == 2500000.0).any()

    def test_employee_dates_preserved(self, employee_df):
        cleaner = AutoClean(employee_df)
        cleaned = cleaner.clean()
        assert pd.api.types.is_datetime64_any_dtype(cleaned["hire_date"])


# ============================================================================
# 2. Customer Dataset Test Suite
# ============================================================================

class TestRealWorldCustomerDataset:
    def test_workflow_runs_and_preserves_input(self, customer_df, capsys):
        snap = customer_df.copy()
        cleaner = AutoClean(customer_df)

        info = cleaner.inspect()
        assert "customer_id" in info["id_like_cols"]

        plan = cleaner.preview()
        assert "country" in plan["inconsistent_categories"]

        cleaned = cleaner.clean()
        cleaner.report()
        cleaner.history()

        pd.testing.assert_frame_equal(customer_df, snap)

    def test_customer_inconsistent_categories_warned_not_modified(self, customer_df):
        cleaner = AutoClean(customer_df)
        plan = cleaner.preview()
        assert "country" in plan["inconsistent_categories"]
        entries = plan["inconsistent_categories"]["country"]
        assert len(entries) >= 1

        cleaned = cleaner.clean()
        # Crucial requirement: inconsistent category values must NOT be changed automatically
        assert "united states" in cleaned["country"].values
        assert "UNITED STATES" in cleaned["country"].values
        assert "United States" in cleaned["country"].values

    def test_customer_id_and_high_cardinality_flagged_not_removed(self, customer_df):
        cleaner = AutoClean(customer_df)
        info = cleaner.inspect()
        assert "customer_id" in info["id_like_cols"]
        assert "city" in info["high_cardinality_cols"]

        cleaned = cleaner.clean()
        # Neither customer_id nor city should be dropped
        assert "customer_id" in cleaned.columns
        assert "city" in cleaned.columns
        assert len(cleaned.columns) == len(customer_df.columns)

    def test_customer_missing_values_imputed(self, customer_df):
        cleaner = AutoClean(customer_df)
        cleaned = cleaner.clean()
        assert cleaned["age"].isnull().sum() == 0
        assert cleaned["annual_spend"].isnull().sum() == 0


# ============================================================================
# 3. Sales Dataset Test Suite
# ============================================================================

class TestRealWorldSalesDataset:
    def test_workflow_runs_and_preserves_input(self, sales_df, capsys):
        snap = sales_df.copy()
        cleaner = AutoClean(sales_df)

        info = cleaner.inspect()
        assert info["duplicates"] == 2

        plan = cleaner.preview()
        assert "quantity" in plan["outliers"]

        cleaned = cleaner.clean()
        cleaner.report()
        cleaner.history()

        pd.testing.assert_frame_equal(sales_df, snap)

    def test_sales_duplicate_transactions_removed(self, sales_df):
        cleaner = AutoClean(sales_df)
        cleaned = cleaner.clean()
        assert len(cleaned) == 10
        assert cleaned.duplicated().sum() == 0

    def test_sales_extreme_values_kept(self, sales_df):
        cleaner = AutoClean(sales_df)
        cleaned = cleaner.clean()
        # Extreme quantity=5000 and total_amount=100000.0 must be preserved
        assert (cleaned["quantity"] == 5000).any()
        assert (cleaned["total_amount"] == 100000.0).any()

    def test_sales_dates_preserved(self, sales_df):
        cleaner = AutoClean(sales_df)
        cleaned = cleaner.clean()
        assert pd.api.types.is_datetime64_any_dtype(cleaned["transaction_date"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned["ship_date"])

    def test_sales_redundant_numerical_detected(self, sales_df):
        cleaner = AutoClean(sales_df)
        info = cleaner.inspect()
        # quantity and total_amount are strongly correlated
        assert len(info["redundant_numerical"]) >= 1


# ============================================================================
# 4. Student Dataset Test Suite
# ============================================================================

class TestRealWorldStudentDataset:
    def test_workflow_runs_and_preserves_input(self, student_df, capsys):
        snap = student_df.copy()
        cleaner = AutoClean(student_df)

        info = cleaner.inspect()
        assert "empty_notes" in info["empty_cols"]
        assert "school_code" in info["constant_cols"]
        assert "academic_year" in info["constant_cols"]

        plan = cleaner.preview()
        assert "empty_notes" in plan["empty_cols_to_drop"]

        cleaned = cleaner.clean()
        cleaner.report()
        cleaner.history()

        pd.testing.assert_frame_equal(student_df, snap)

    def test_student_empty_column_dropped(self, student_df):
        cleaner = AutoClean(student_df)
        cleaned = cleaner.clean()
        assert "empty_notes" not in cleaned.columns

    def test_student_constant_columns_kept(self, student_df):
        cleaner = AutoClean(student_df)
        cleaned = cleaner.clean()
        # Constant columns must ONLY be flagged, never automatically removed
        assert "school_code" in cleaned.columns
        assert "academic_year" in cleaned.columns
        assert (cleaned["school_code"] == "SCH-101").all()
        assert (cleaned["academic_year"] == 2026).all()

    def test_student_missing_values_imputed(self, student_df):
        cleaner = AutoClean(student_df)
        cleaned = cleaner.clean()
        assert cleaned.isnull().sum().sum() == 0
        # Categorical major imputed
        assert cleaned["major"].isnull().sum() == 0
        # Numerical gpa & attendance_pct imputed
        assert cleaned["gpa"].isnull().sum() == 0
        assert cleaned["attendance_pct"].isnull().sum() == 0

    def test_student_history_and_report_accuracy(self, student_df):
        cleaner = AutoClean(student_df)
        cleaner.clean()
        hist = cleaner.history()

        operations = [entry["operation"] for entry in hist]
        assert "drop_empty_cols" in operations
        assert "impute_categorical" in operations
        assert "impute_numerical" in operations

        summary = cleaner._last_clean_summary
        assert summary["rows_before"] == 10
        assert summary["rows_after"] == 10
        assert summary["cols_before"] == 8
        assert summary["cols_after"] == 7
        assert "empty_notes" in summary["empty_cols_dropped"]
