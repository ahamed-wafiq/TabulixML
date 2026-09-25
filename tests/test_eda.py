"""
tests/test_eda.py -- Comprehensive pytest suite for datacraft.AutoEDA

Tests:
  - Instantiation & validation (non-DataFrame, empty DataFrame)
  - inspect(): shape, rows, columns, data types, missing values, unique counts, duplicates
  - summary(): numerical stats (count, mean, median, std, min, max),
               categorical stats (unique, top, freq), single-column summary
  - correlations(): numerical correlation matrix, threshold filtering, high-correlation pairs
  - quality(): missing rates, duplicate counts, constant columns, high cardinality, outliers
  - report(): formatted text output and sections
  - Immutability guarantee: original DataFrame is NEVER modified
"""

import os
import numpy as np
import pandas as pd
import pytest
from datacraft import AutoEDA


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Standard mixed-type DataFrame for EDA testing."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Alice"],
        "age": [25.0, 30.0, np.nan, 45.0, 25.0],
        "salary": [50000.0, 60000.0, 55000.0, 2500000.0, 50000.0],
        "department": ["Engineering", "Sales", "HR", "Sales", "Engineering"],
        "joined": pd.to_datetime(["2020-01-15", "2019-06-01", "2021-03-22", "2018-11-05", "2020-01-15"]),
        "status": ["Active", "Active", "Active", "Active", "Active"],  # Constant
    })


@pytest.fixture()
def numerical_only_df() -> pd.DataFrame:
    """Numerical-only DataFrame with known correlation."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfect correlation (r = 1.0)
    z = [5.0, 4.0, 2.0, 1.0, 0.0]  # Negative correlation (r = -0.99)
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture()
def categorical_only_df() -> pd.DataFrame:
    """Categorical-only DataFrame."""
    return pd.DataFrame({
        "color": ["red", "blue", "red", "green", "blue", "red"],
        "size": ["S", "M", "L", "S", "M", "L"],
    })


# ============================================================================
# 1. Instantiation and Validation
# ============================================================================

class TestEDAInstantiation:
    def test_valid_instantiation(self, sample_df):
        eda = AutoEDA(sample_df)
        assert eda is not None
        assert eda.dataframe.shape == sample_df.shape

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            AutoEDA([1, 2, 3])

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            AutoEDA({"a": [1, 2]})

    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            AutoEDA(pd.DataFrame())

        with pytest.raises(ValueError, match="empty"):
            AutoEDA(pd.DataFrame({"a": [], "b": []}))


# ============================================================================
# 2. inspect()
# ============================================================================

class TestEDAInspect:
    def test_inspect_shape_and_counts(self, sample_df):
        eda = AutoEDA(sample_df)
        info = eda.inspect()

        assert info["shape"] == (5, 6)
        assert info["rows"] == 5
        assert info["column_count"] == 6
        assert set(info["columns"]) == set(sample_df.columns)

    def test_inspect_missing_values(self, sample_df):
        eda = AutoEDA(sample_df)
        info = eda.inspect()

        assert info["total_missing"] == 1
        assert info["missing"]["age"]["count"] == 1
        assert info["missing"]["age"]["percentage"] == 20.0
        assert info["missing"]["salary"]["count"] == 0

    def test_inspect_duplicates(self, sample_df):
        eda = AutoEDA(sample_df)
        info = eda.inspect()

        # Row 0 and Row 4 are identical: ("Alice", 25.0, 50000.0, "Engineering", "2020-01-15", "Active")
        assert info["duplicates"] == 1
        assert info["duplicate_percentage"] == 20.0

    def test_inspect_unique_counts(self, sample_df):
        eda = AutoEDA(sample_df)
        info = eda.inspect()

        assert info["unique_counts"]["department"] == 3
        assert info["unique_counts"]["status"] == 1

    def test_inspect_detected_types(self, sample_df):
        eda = AutoEDA(sample_df)
        info = eda.inspect()

        assert info["col_types"]["age"] == "numerical"
        assert info["col_types"]["salary"] == "numerical"
        assert info["col_types"]["department"] == "categorical"
        assert info["col_types"]["joined"] == "datetime"


# ============================================================================
# 3. summary()
# ============================================================================

class TestEDASummary:
    def test_numerical_summary_metrics(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.summary()

        assert "numerical" in res
        num_df = res["numerical"]
        assert isinstance(num_df, pd.DataFrame)
        assert "salary" in num_df.index
        assert "age" in num_df.index

        # Check required columns
        for col in ["count", "mean", "median", "std", "min", "max"]:
            assert col in num_df.columns

        # Verify specific stats for age (values: 25, 30, 45, 25; non-null count: 4)
        assert num_df.loc["age", "count"] == 4
        assert num_df.loc["age", "min"] == 25.0
        assert num_df.loc["age", "max"] == 45.0
        assert num_df.loc["age", "median"] == 27.5

    def test_categorical_summary_metrics(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.summary()

        assert "categorical" in res
        cat_df = res["categorical"]
        assert isinstance(cat_df, pd.DataFrame)
        assert "department" in cat_df.index
        assert "status" in cat_df.index

        # Check required columns: unique, top, freq
        for col in ["unique", "top", "freq"]:
            assert col in cat_df.columns

        assert cat_df.loc["status", "unique"] == 1
        assert cat_df.loc["status", "top"] == "Active"
        assert cat_df.loc["status", "freq"] == 5

    def test_single_column_summary(self, sample_df):
        eda = AutoEDA(sample_df)
        age_series = eda.summary("age")
        assert isinstance(age_series, pd.Series)
        assert age_series["count"] == 4
        assert "mean" in age_series

        dept_series = eda.summary("department")
        assert isinstance(dept_series, pd.Series)
        assert dept_series["unique"] == 3
        assert "top" in dept_series

    def test_single_column_not_found_raises(self, sample_df):
        eda = AutoEDA(sample_df)
        with pytest.raises(ValueError, match="not found"):
            eda.summary("non_existent_column")

    def test_summary_no_numerical_columns(self, categorical_only_df):
        eda = AutoEDA(categorical_only_df)
        res = eda.summary()
        assert res["numerical"].empty
        assert not res["categorical"].empty

    def test_summary_no_categorical_columns(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        res = eda.summary()
        assert not res["numerical"].empty
        assert res["categorical"].empty


# ============================================================================
# 4. correlations()
# ============================================================================

class TestEDACorrelations:
    def test_correlations_matrix_and_pairs(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        res = eda.correlations(threshold=0.90)

        assert "matrix" in res
        assert "high_correlations" in res
        matrix = res["matrix"]
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (3, 3)

        # Perfect correlation between x and y (r = 1.0)
        assert np.isclose(matrix.loc["x", "y"], 1.0)

        # High correlations list should contain (x, y) and (x, z) / (y, z)
        high = res["high_correlations"]
        pair_names = [(h["col1"], h["col2"]) for h in high]
        assert ("x", "y") in pair_names

    def test_correlations_threshold_filtering(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        # With threshold 0.999, only x and y (r = 1.0) should match
        res = eda.correlations(threshold=0.999)
        assert len(res["high_correlations"]) == 1
        assert res["high_correlations"][0]["col1"] == "x"
        assert res["high_correlations"][0]["col2"] == "y"

    def test_correlations_fewer_than_two_numerical_cols(self, categorical_only_df):
        eda = AutoEDA(categorical_only_df)
        res = eda.correlations()
        assert res["high_correlations"] == []
        assert res["matrix"].empty

    def test_correlations_does_not_remove_columns(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        cols_before = list(numerical_only_df.columns)
        eda.correlations(threshold=0.5)
        # Underlying dataframe must retain all original columns
        assert list(eda.dataframe.columns) == cols_before


# ============================================================================
# 5. quality()
# ============================================================================

class TestEDAQuality:
    def test_quality_metrics_detection(self, sample_df):
        eda = AutoEDA(sample_df)
        q = eda.quality()

        assert "missing_percentage" in q
        assert "duplicate_count" in q
        assert "constant_columns" in q
        assert "high_cardinality_columns" in q
        assert "outlier_counts" in q

        # Missing percentage: 1 missing cell out of 30 total cells = 3.33%
        assert q["total_missing"] == 1
        assert q["missing_percentage"] > 0
        assert "age" in q["missing_by_column"]

        # Duplicate count: 1 duplicate row
        assert q["duplicate_count"] == 1

        # Constant column: 'status'
        assert "status" in q["constant_columns"]

        # Outlier detection: 'salary' has an extreme outlier (2,500,000)
        assert "salary" in q["outlier_counts"]
        assert q["outlier_counts"]["salary"] >= 1

    def test_quality_clean_dataset(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        q = eda.quality()

        assert q["total_missing"] == 0
        assert q["missing_percentage"] == 0.0
        assert q["duplicate_count"] == 0
        assert q["constant_columns"] == []
        assert q["total_outliers"] == 0


# ============================================================================
# 6. report()
# ============================================================================

class TestEDAReport:
    def test_report_prints_and_returns_string(self, sample_df, capsys):
        eda = AutoEDA(sample_df)
        rep = eda.report(print_report=True)

        assert isinstance(rep, str)
        assert "DataCraft -- AutoEDA Report" in rep
        assert "Dataset Shape" in rep
        assert "Numerical Features Summary" in rep
        assert "Categorical Features Summary" in rep
        assert "Data Quality Findings" in rep

        # Verify stdout
        captured = capsys.readouterr().out
        assert "DataCraft -- AutoEDA Report" in captured

    def test_report_print_false_suppresses_stdout(self, sample_df, capsys):
        eda = AutoEDA(sample_df)
        rep = eda.report(print_report=False)
        assert isinstance(rep, str)
        captured = capsys.readouterr().out
        assert captured == ""


# ============================================================================
# 7. Immutability Guarantee
# ============================================================================

class TestEDAImmutability:
    def test_eda_never_mutates_original_dataframe(self, sample_df, tmp_path):
        snapshot = sample_df.copy()
        eda = AutoEDA(sample_df)

        eda.inspect()
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.summary()
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.correlations()
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.quality()
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.report(print_report=False)
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.visualize()
        pd.testing.assert_frame_equal(sample_df, snapshot)

        eda.save_report(str(tmp_path / "immutability_report.html"))
        pd.testing.assert_frame_equal(sample_df, snapshot)


# ============================================================================
# 8. visualize()
# ============================================================================

class TestEDAVisualize:
    def test_visualize_basic(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.visualize()

        assert isinstance(res, dict)
        assert "numerical" in res
        assert "categorical" in res
        assert "correlations" in res
        assert "plots_count" in res
        assert res["plots_count"] > 0
        assert res["saved_files"] == []

    def test_visualize_numerical_plots(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.visualize()

        assert "age" in res["numerical"]
        assert "salary" in res["numerical"]
        assert "histogram" in res["numerical"]["age"]
        assert "boxplot" in res["numerical"]["age"]
        assert "histogram_base64" in res["numerical"]["age"]
        assert "boxplot_base64" in res["numerical"]["age"]

    def test_visualize_categorical_plots(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.visualize()

        assert "department" in res["categorical"]
        assert "bar_chart" in res["categorical"]["department"]
        assert "bar_chart_base64" in res["categorical"]["department"]

    def test_visualize_correlation_heatmap(self, sample_df):
        eda = AutoEDA(sample_df)
        res = eda.visualize()

        # sample_df has age and salary (2 numerical features)
        assert "heatmap" in res["correlations"]
        assert "heatmap_base64" in res["correlations"]

    def test_visualize_output_dir_creates_files(self, sample_df, tmp_path):
        out_dir = tmp_path / "eda_output"
        eda = AutoEDA(sample_df)
        res = eda.visualize(output_dir=str(out_dir))

        assert out_dir.exists()
        assert len(res["saved_files"]) > 0
        for fpath in res["saved_files"]:
            assert os.path.exists(fpath)
            assert os.path.getsize(fpath) > 0

    def test_visualize_no_numerical_columns(self, categorical_only_df):
        eda = AutoEDA(categorical_only_df)
        res = eda.visualize()

        assert res["numerical"] == {}
        assert res["correlations"] == {}
        assert len(res["categorical"]) > 0
        assert "color" in res["categorical"]

    def test_visualize_no_categorical_columns(self, numerical_only_df):
        eda = AutoEDA(numerical_only_df)
        res = eda.visualize()

        assert len(res["numerical"]) > 0
        assert res["categorical"] == {}
        assert "heatmap" in res["correlations"]

    def test_visualize_with_missing_values(self):
        df_missing = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, "cat", "dog", np.nan, "cat"],
        })
        eda = AutoEDA(df_missing)
        res = eda.visualize()

        assert "a" in res["numerical"]
        assert "b" in res["categorical"]
        assert res["plots_count"] > 0

    def test_visualize_single_value_and_constant_columns(self):
        df_const = pd.DataFrame({
            "num_const": [42.0, 42.0, 42.0, 42.0],
            "cat_const": ["fixed", "fixed", "fixed", "fixed"],
        })
        eda = AutoEDA(df_const)
        res = eda.visualize()

        assert "num_const" in res["numerical"]
        assert "cat_const" in res["categorical"]
        assert res["plots_count"] > 0

    def test_visualize_all_nan_column(self):
        df_empty_col = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "empty_num": [np.nan, np.nan, np.nan],
            "empty_cat": [None, None, None],
        })
        eda = AutoEDA(df_empty_col)
        res = eda.visualize()

        assert "num" in res["numerical"]
        # Empty columns are safely skipped from plotting
        assert "empty_num" not in res["numerical"]

    def test_visualize_small_dataset(self):
        # 1-row DataFrame
        df_one = pd.DataFrame({"x": [10.0], "label": ["yes"]})
        eda_one = AutoEDA(df_one)
        res_one = eda_one.visualize()
        assert res_one["plots_count"] > 0

        # 2-row DataFrame
        df_two = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "label": ["yes", "no"]})
        eda_two = AutoEDA(df_two)
        res_two = eda_two.visualize()
        assert res_two["plots_count"] > 0

    def test_visualize_many_columns_safe_limit(self):
        data = {f"col_{i}": np.random.randn(20) for i in range(25)}
        df_many = pd.DataFrame(data)
        eda = AutoEDA(df_many)
        res = eda.visualize(max_cols=5)

        # Only 5 columns should have plots generated
        assert len(res["numerical"]) == 5


# ============================================================================
# 9. save_report()
# ============================================================================

class TestEDASaveReport:
    def test_save_report_creates_file(self, sample_df, tmp_path):
        report_file = tmp_path / "eda_report.html"
        eda = AutoEDA(sample_df)
        saved_path = eda.save_report(str(report_file))

        assert os.path.exists(saved_path)
        assert os.path.getsize(saved_path) > 0

    def test_save_report_standalone_html_contents(self, sample_df, tmp_path):
        report_file = tmp_path / "report.html"
        eda = AutoEDA(sample_df)
        eda.save_report(str(report_file), title="Test Analysis Report")

        content = report_file.read_text(encoding="utf-8")

        # HTML Structure
        assert "<!DOCTYPE html>" in content
        assert "Test Analysis Report" in content
        assert "<style>" in content
        assert "</html>" in content

        # Required components
        assert "5 rows" in content  # dataset shape
        assert "6 columns" in content
        assert "Dataset Schema &amp; Column Types" in content or "Dataset Schema" in content  # column types
        assert "Missing Values" in content  # missing values
        assert "Unique Values" in content  # unique values
        assert "Numerical Features Summary" in content  # numerical summary
        assert "Categorical Features Summary" in content  # categorical summary
        assert "Feature Correlations" in content  # correlations
        assert "Data-Quality Findings &amp; Warnings" in content or "Data Quality" in content  # quality warnings
        assert "Numerical Distributions" in content  # visualizations
        assert "Categorical Distributions" in content
        assert "data:image/png;base64," in content  # embedded visualizations

    def test_save_report_without_calling_visualize_first(self, sample_df, tmp_path):
        report_file = tmp_path / "auto_vis_report.html"
        eda = AutoEDA(sample_df)
        assert eda._visualizations is None

        eda.save_report(str(report_file))

        assert eda._visualizations is not None
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_save_report_no_numerical_columns(self, categorical_only_df, tmp_path):
        report_file = tmp_path / "cat_report.html"
        eda = AutoEDA(categorical_only_df)
        eda.save_report(str(report_file))

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "No numerical columns detected" in content

    def test_save_report_no_categorical_columns(self, numerical_only_df, tmp_path):
        report_file = tmp_path / "num_report.html"
        eda = AutoEDA(numerical_only_df)
        eda.save_report(str(report_file))

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "No categorical columns detected" in content

    def test_save_report_with_missing_values(self, tmp_path):
        df_missing = pd.DataFrame({
            "age": [20.0, np.nan, 30.0],
            "city": ["Paris", None, "London"],
        })
        report_file = tmp_path / "missing_report.html"
        eda = AutoEDA(df_missing)
        eda.save_report(str(report_file))

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "Missing Values Detected" in content

    def test_save_report_small_dataset(self, tmp_path):
        df_small = pd.DataFrame({"val": [100.0]})
        report_file = tmp_path / "small_report.html"
        eda = AutoEDA(df_small)
        eda.save_report(str(report_file))

        assert report_file.exists()
        assert report_file.stat().st_size > 0

    def test_save_report_creates_parent_directories(self, sample_df, tmp_path):
        nested_file = tmp_path / "sub" / "folder" / "nested_report.html"
        eda = AutoEDA(sample_df)
        eda.save_report(str(nested_file))

        assert nested_file.exists()

