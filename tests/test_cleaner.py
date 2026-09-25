"""
tests/test_cleaner.py -- Comprehensive pytest suite for datacraft.AutoClean

Covers:
  Realistic messy DataFrame fixture, column-type detection,
  missing-value detection and imputation (mean/median/mode/auto),
  duplicate removal, empty and constant column handling,
  IQR outlier flagging (never removed), column-name cleaning,
  cleaning history, report() before/after output, immutability.
"""

import numpy as np
import pandas as pd
import pytest

from datacraft import AutoClean


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def messy_df() -> pd.DataFrame:
    """
    Realistic messy DataFrame exercising every cleaning feature.

    age         : numerical, 2 missing
    salary      : numerical, 1 missing, outlier value 999
    department  : categorical, 1 missing, inconsistent casing
    hire_date   : datetime, complete
    empty_col   : 100 % NaN -- dropped by clean()
    constant_col: constant 0.0 -- flagged only, kept
    Rows 9-10 duplicate rows 0-1.
    """
    data = {
        "age": [25.0, 30.0, np.nan, 40.0, 35.0, 28.0, 45.0, 32.0, np.nan, 25.0, 30.0],
        "salary": [
            50000.0, 60000.0, 55000.0, np.nan,
            70000.0, 52000.0, 999.0,
            65000.0, 58000.0, 50000.0, 60000.0,
        ],
        "department": [
            "Engineering", "Sales", "HR", "Engineering",
            "engineering", "ENGINEERING", np.nan,
            "Sales", "HR", "Engineering", "Sales",
        ],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2019-07-01", "2021-03-22", "2018-11-05",
            "2022-05-10", "2023-01-30", "2017-09-14",
            "2020-08-19", "2016-04-03", "2020-01-15", "2019-07-01",
        ]),
        "empty_col":    [np.nan] * 11,
        "constant_col": [0.0]   * 11,
    }
    return pd.DataFrame(data)


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":    [25, 30, np.nan, 40, 30, 30],
        "salary": [50000.0, 60000.0, 55000.0, np.nan, 60000.0, 60000.0],
        "city":   ["London", "Paris", np.nan, "Berlin", "Paris", "Paris"],
        "joined": pd.to_datetime([
            "2020-01-01", "2021-06-15", "2019-03-10",
            "2022-11-01", "2021-06-15", "2021-06-15",
        ]),
    })


@pytest.fixture()
def df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 2, 3], "y": ["a", "b", "b", "c"]})


@pytest.fixture()
def df_with_empty_col() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1, 2, 3],
        "empty": [np.nan, np.nan, np.nan],
        "b": ["x", "y", "z"],
    })


@pytest.fixture()
def df_with_constant_col() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "const": ["same", "same", "same"]})


@pytest.fixture()
def df_with_outliers() -> pd.DataFrame:
    return pd.DataFrame({"score": [10, 11, 12, 10, 11, 9, 10, 1000]})


# ===========================================================================
# 1. Instantiation
# ===========================================================================

class TestInstantiation:
    def test_accepts_dataframe(self, simple_df):
        assert AutoClean(simple_df) is not None

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            AutoClean([[1, 2], [3, 4]])

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError):
            AutoClean(pd.DataFrame())

    def test_default_parameters(self, simple_df):
        c = AutoClean(simple_df)
        assert c._num_strategy == "median"
        assert c._clean_col_names is False
        assert c._iqr_threshold == 1.5

    def test_custom_parameters(self, simple_df):
        c = AutoClean(simple_df, num_strategy="mean", clean_col_names=True, iqr_threshold=3.0)
        assert c._num_strategy == "mean"
        assert c._clean_col_names is True
        assert c._iqr_threshold == 3.0


# ===========================================================================
# 2. Immutability -- original DataFrame must never be mutated
# ===========================================================================

class TestImmutability:
    def test_original_not_mutated_after_clean(self, simple_df):
        snap = simple_df.copy()
        AutoClean(simple_df).clean()
        pd.testing.assert_frame_equal(simple_df, snap)

    def test_original_not_mutated_after_inspect(self, simple_df):
        snap = simple_df.copy()
        AutoClean(simple_df).inspect()
        pd.testing.assert_frame_equal(simple_df, snap)

    def test_messy_original_not_mutated(self, messy_df):
        snap = messy_df.copy()
        AutoClean(messy_df).clean()
        pd.testing.assert_frame_equal(messy_df, snap)

    def test_clean_returns_new_object(self, simple_df):
        cleaner = AutoClean(simple_df)
        result = cleaner.clean()
        assert result is not simple_df
        assert result is not cleaner._original


# ===========================================================================
# 3. Column-type detection
# ===========================================================================

class TestColumnTypeDetection:
    def test_numerical_cols_detected(self, messy_df):
        info = AutoClean(messy_df).inspect()
        assert info["col_types"]["age"] == "numerical"
        assert info["col_types"]["salary"] == "numerical"

    def test_categorical_col_detected(self, messy_df):
        assert AutoClean(messy_df).inspect()["col_types"]["department"] == "categorical"

    def test_datetime_col_detected(self, messy_df):
        assert AutoClean(messy_df).inspect()["col_types"]["hire_date"] == "datetime"

    def test_simple_df_types(self, simple_df):
        info = AutoClean(simple_df).inspect()
        assert info["col_types"]["age"] == "numerical"
        assert info["col_types"]["city"] == "categorical"
        assert info["col_types"]["joined"] == "datetime"


# ===========================================================================
# 4. Missing-value detection
# ===========================================================================

class TestMissingValueDetection:
    def test_correct_missing_counts_messy(self, messy_df):
        info = AutoClean(messy_df).inspect()
        assert info["missing"]["age"] == 2
        assert info["missing"]["salary"] == 1
        assert info["missing"]["department"] == 1

    def test_empty_col_reported_as_missing(self, messy_df):
        info = AutoClean(messy_df).inspect()
        assert info["missing"]["empty_col"] == 11

    def test_no_missing_key_when_complete(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert AutoClean(df).inspect()["missing"] == {}

    def test_missing_counts_simple_df(self, simple_df):
        info = AutoClean(simple_df).inspect()
        assert info["missing"]["age"] == 1
        assert info["missing"]["salary"] == 1
        assert info["missing"]["city"] == 1


# ===========================================================================
# 5. Numerical imputation -- mean / median / auto strategies
# ===========================================================================

class TestNumericalImputation:
    def test_median_strategy_correct_value(self):
        # median([1,2,3]) = 2.0
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, np.nan]})
        cleaned = AutoClean(df, num_strategy="median").clean()
        assert cleaned["v"].isnull().sum() == 0
        assert cleaned["v"].iloc[-1] == pytest.approx(2.0)

    def test_mean_strategy_correct_value(self):
        # mean([1,3]) = 2.0
        df = pd.DataFrame({"v": [1.0, 3.0, np.nan]})
        cleaned = AutoClean(df, num_strategy="mean").clean()
        assert cleaned["v"].isnull().sum() == 0
        assert cleaned["v"].iloc[-1] == pytest.approx(2.0)

    def test_auto_skewed_selects_median(self):
        values = [1.0] * 10 + [500.0, np.nan]
        df = pd.DataFrame({"v": values})
        auto   = AutoClean(df, num_strategy="auto").clean()
        median = AutoClean(df, num_strategy="median").clean()
        assert auto["v"].iloc[-1] == pytest.approx(median["v"].iloc[-1])

    def test_auto_symmetric_selects_mean(self):
        values = [10.0, 10.0, 10.0, 10.0, np.nan]
        df = pd.DataFrame({"v": values})
        auto = AutoClean(df, num_strategy="auto").clean()
        mean = AutoClean(df, num_strategy="mean").clean()
        assert auto["v"].iloc[-1] == pytest.approx(mean["v"].iloc[-1])

    def test_no_numerical_missing_after_clean(self, messy_df):
        cleaned = AutoClean(messy_df).clean()
        assert cleaned["age"].isnull().sum() == 0
        assert cleaned["salary"].isnull().sum() == 0

    def test_imputation_recorded_in_history(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        steps = [e["step"] for e in cleaner.history()]
        assert "impute_numerical" in steps

    def test_history_detail_has_fill_value(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        imp = [e for e in cleaner.history() if e["step"] == "impute_numerical"]
        assert imp
        for entry in imp:
            assert "fill_value" in entry["details"]
            assert "missing_count" in entry["details"]
            assert entry["details"]["missing_count"] > 0


# ===========================================================================
# 6. Categorical imputation -- mode fill
# ===========================================================================

class TestCategoricalImputation:
    def test_mode_fill_correct_value(self):
        df = pd.DataFrame({"cat": ["a", "a", "a", "b", np.nan]})
        cleaned = AutoClean(df).clean()
        assert cleaned["cat"].isnull().sum() == 0
        assert cleaned["cat"].iloc[-1] == "a"

    def test_no_categorical_missing_after_clean(self, simple_df):
        assert AutoClean(simple_df).clean()["city"].isnull().sum() == 0

    def test_mode_selects_most_frequent(self):
        # Each row is unique (different 'id') so no deduplication occurs.
        # Paris appears 3x so mode = "Paris".
        df = pd.DataFrame({
            "id":   [1, 2, 3, 4, 5, 6],
            "city": ["London", "Paris", "Paris", "Paris", "Berlin", np.nan],
        })
        cleaned = AutoClean(df).clean()
        # The NaN was at index 5; after clean it must be filled with "Paris"
        assert cleaned["city"].isnull().sum() == 0
        assert cleaned.loc[cleaned["id"] == 6, "city"].iloc[0] == "Paris"

    def test_messy_department_imputed(self, messy_df):
        assert AutoClean(messy_df).clean()["department"].isnull().sum() == 0

    def test_categorical_imputation_recorded_in_history(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        assert "impute_categorical" in [e["step"] for e in cleaner.history()]


# ===========================================================================
# 7. Duplicate row removal
# ===========================================================================

class TestDuplicateRemoval:
    def test_duplicates_detected_in_inspect(self, df_with_duplicates):
        assert AutoClean(df_with_duplicates).inspect()["duplicates"] == 1

    def test_duplicates_removed_in_clean(self, df_with_duplicates):
        assert len(AutoClean(df_with_duplicates).clean()) == 3

    def test_no_duplicates_remain_after_clean(self, df_with_duplicates):
        assert AutoClean(df_with_duplicates).clean().duplicated().sum() == 0

    def test_messy_df_duplicates_detected(self, messy_df):
        assert AutoClean(messy_df).inspect()["duplicates"] == 2

    def test_messy_df_duplicates_removed(self, messy_df):
        assert AutoClean(messy_df).clean().duplicated().sum() == 0

    def test_duplicate_removal_recorded_in_history(self, df_with_duplicates):
        cleaner = AutoClean(df_with_duplicates)
        cleaner.clean()
        assert "drop_duplicates" in [e["step"] for e in cleaner.history()]

    def test_no_dup_step_logged_when_none(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        cleaner = AutoClean(df)
        cleaner.clean()
        assert [e for e in cleaner.history() if e["step"] == "drop_duplicates"] == []

    def test_zero_dup_count_for_clean_data(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        assert AutoClean(df).inspect()["duplicates"] == 0


# ===========================================================================
# 8. Empty-column detection and removal
# ===========================================================================

class TestEmptyColumns:
    def test_empty_col_detected(self, df_with_empty_col):
        assert "empty" in AutoClean(df_with_empty_col).inspect()["empty_cols"]

    def test_empty_col_removed_by_clean(self, df_with_empty_col):
        assert "empty" not in AutoClean(df_with_empty_col).clean().columns

    def test_non_empty_cols_retained(self, df_with_empty_col):
        cleaned = AutoClean(df_with_empty_col).clean()
        assert "a" in cleaned.columns
        assert "b" in cleaned.columns

    def test_messy_empty_col_detected(self, messy_df):
        assert "empty_col" in AutoClean(messy_df).inspect()["empty_cols"]

    def test_messy_empty_col_removed_by_clean(self, messy_df):
        assert "empty_col" not in AutoClean(messy_df).clean().columns

    def test_empty_col_removal_recorded_in_history(self, df_with_empty_col):
        cleaner = AutoClean(df_with_empty_col)
        cleaner.clean()
        assert "drop_empty_cols" in [e["step"] for e in cleaner.history()]

    def test_no_empty_cols_for_complete_data(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        assert AutoClean(df).inspect()["empty_cols"] == []


# ===========================================================================
# 9. Constant-column detection -- flagged, NOT removed
# ===========================================================================

class TestConstantColumns:
    def test_constant_col_detected(self, df_with_constant_col):
        assert "const" in AutoClean(df_with_constant_col).inspect()["constant_cols"]

    def test_non_constant_not_flagged(self, df_with_constant_col):
        assert "a" not in AutoClean(df_with_constant_col).inspect()["constant_cols"]

    def test_messy_constant_col_detected(self, messy_df):
        assert "constant_col" in AutoClean(messy_df).inspect()["constant_cols"]

    def test_constant_col_NOT_removed_by_clean(self, messy_df):
        assert "constant_col" in AutoClean(messy_df).clean().columns

    def test_numerical_constant_detected(self):
        df = pd.DataFrame({"v": [7, 7, 7, 7]})
        assert "v" in AutoClean(df).inspect()["constant_cols"]

    def test_no_constant_for_varied_data(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert AutoClean(df).inspect()["constant_cols"] == []


# ===========================================================================
# 10. IQR outlier detection -- flagged only, NEVER removed
# ===========================================================================

class TestOutlierDetection:
    def test_outlier_flagged_in_inspect(self, df_with_outliers):
        assert AutoClean(df_with_outliers).inspect()["outliers"]["score"]["count"] >= 1

    def test_outlier_value_present_after_clean(self, df_with_outliers):
        assert 1000 in AutoClean(df_with_outliers).clean()["score"].values

    def test_fence_values_in_outlier_info(self, df_with_outliers):
        o = AutoClean(df_with_outliers).inspect()["outliers"]["score"]
        assert "lower_fence" in o
        assert "upper_fence" in o
        assert "q1" in o
        assert "q3" in o

    def test_outlier_count_is_positive(self, df_with_outliers):
        assert AutoClean(df_with_outliers).inspect()["outliers"]["score"]["count"] > 0

    def test_no_outlier_count_for_normal_data(self):
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6]})
        assert AutoClean(df).inspect()["outliers"].get("v", {}).get("count", 0) == 0

    def test_messy_salary_outlier_flagged(self, messy_df):
        info = AutoClean(messy_df).inspect()
        assert "salary" in info["outliers"]
        assert info["outliers"]["salary"]["count"] >= 1

    def test_messy_salary_outlier_not_deleted(self, messy_df):
        assert 999.0 in AutoClean(messy_df).clean()["salary"].values

    def test_iqr_threshold_affects_count(self):
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 20]})
        strict_count = AutoClean(df, iqr_threshold=1.5).inspect()["outliers"].get("v", {}).get("count", 0)
        loose_count  = AutoClean(df, iqr_threshold=3.0).inspect()["outliers"].get("v", {}).get("count", 0)
        assert loose_count <= strict_count


# ===========================================================================
# 11. Column-name cleaning
# ===========================================================================

class TestColumnNameCleaning:
    def test_spaces_replaced_with_underscore(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        cleaned = AutoClean(df, clean_col_names=True).clean()
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns

    def test_special_chars_stripped(self):
        df = pd.DataFrame({"$Price!": [1.0], "Count#": [2]})
        cleaned = AutoClean(df, clean_col_names=True).clean()
        assert "price" in cleaned.columns
        assert "count" in cleaned.columns

    def test_duplicate_col_names_deduplicated(self):
        df = pd.DataFrame([[1, 2]], columns=["col A", "col A"])
        cleaned = AutoClean(df, clean_col_names=True).clean()
        assert len(set(cleaned.columns)) == len(cleaned.columns)

    def test_original_names_preserved_when_disabled(self, simple_df):
        assert "age" in AutoClean(simple_df, clean_col_names=False).clean().columns

    def test_clean_names_recorded_in_history(self):
        df = pd.DataFrame({"First Name": [1]})
        cleaner = AutoClean(df, clean_col_names=True)
        cleaner.clean()
        assert "clean_col_names" in [e["step"] for e in cleaner.history()]

    def test_leading_trailing_whitespace_stripped(self):
        df = pd.DataFrame({"  gap  ": [1, 2]})
        assert "gap" in AutoClean(df, clean_col_names=True).clean().columns


# ===========================================================================
# 12. Cleaning history
# ===========================================================================

class TestHistory:
    def test_history_empty_before_any_call(self, simple_df):
        assert AutoClean(simple_df).history() == []

    def test_history_populated_after_clean(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        assert len(cleaner.history()) > 0

    def test_history_populated_after_inspect(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.inspect()
        assert len(cleaner.history()) > 0

    def test_history_entries_have_required_keys(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        for entry in cleaner.history():
            assert "step" in entry
            assert "timestamp" in entry
            assert "message" in entry
            assert "details" in entry

    def test_history_returns_copy_not_reference(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        n = len(cleaner.history())
        h = cleaner.history()
        h.clear()
        assert len(cleaner.history()) == n

    def test_history_timestamps_are_iso_strings(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        for entry in cleaner.history():
            assert isinstance(entry["timestamp"], str)
            assert "T" in entry["timestamp"]

    def test_history_dup_removal_detail(self, df_with_duplicates):
        cleaner = AutoClean(df_with_duplicates)
        cleaner.clean()
        dup = [e for e in cleaner.history() if e["step"] == "drop_duplicates"]
        assert len(dup) == 1
        assert dup[0]["details"]["removed"] == 1

    def test_history_imputation_detail_fields(self, simple_df):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        imp = [e for e in cleaner.history() if "impute" in e["step"]]
        assert imp
        for entry in imp:
            assert "col" in entry["details"]
            assert "fill_value" in entry["details"]
            assert "missing_count" in entry["details"]

    def test_all_expected_steps_in_history(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.clean()
        steps = {e["step"] for e in cleaner.history()}
        assert "inspect"            in steps
        assert "drop_empty_cols"    in steps
        assert "drop_duplicates"    in steps
        assert "impute_numerical"   in steps
        assert "impute_categorical" in steps


# ===========================================================================
# 13. report() -- before/after output
# ===========================================================================

class TestReport:
    def test_report_runs_without_error(self, simple_df, capsys):
        AutoClean(simple_df).report()
        assert "AutoClean Report" in capsys.readouterr().out

    def test_report_before_clean_shows_before_hint(self, simple_df, capsys):
        AutoClean(simple_df).report()
        assert "before" in capsys.readouterr().out.lower()

    def test_report_after_clean_shows_before_after_delta(self, simple_df, capsys):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        cleaner.report()
        out = capsys.readouterr().out
        assert "BEFORE" in out
        assert "AFTER"  in out
        assert "DELTA"  in out

    def test_report_shows_imputed_columns(self, simple_df, capsys):
        cleaner = AutoClean(simple_df)
        cleaner.clean()
        cleaner.report()
        out = capsys.readouterr().out
        assert "age" in out or "salary" in out

    def test_report_shows_duplicate_removal(self, df_with_duplicates, capsys):
        cleaner = AutoClean(df_with_duplicates)
        cleaner.clean()
        cleaner.report()
        assert "duplicate" in capsys.readouterr().out.lower()

    def test_report_shows_empty_col_drop(self, df_with_empty_col, capsys):
        cleaner = AutoClean(df_with_empty_col)
        cleaner.clean()
        cleaner.report()
        assert "empty" in capsys.readouterr().out.lower()

    def test_report_shows_outlier_section(self, df_with_outliers, capsys):
        cleaner = AutoClean(df_with_outliers)
        cleaner.clean()
        cleaner.report()
        assert "outlier" in capsys.readouterr().out.lower()

    def test_report_no_changes_message_for_clean_data(self, capsys):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        cleaner = AutoClean(df)
        cleaner.clean()
        cleaner.report()
        assert "no changes" in capsys.readouterr().out.lower()

    def test_report_messy_df_full(self, messy_df, capsys):
        cleaner = AutoClean(messy_df)
        cleaner.clean()
        cleaner.report()
        out = capsys.readouterr().out
        assert "BEFORE"    in out
        assert "AFTER"     in out
        assert "DELTA"     in out
        assert "empty_col" in out
        assert "outlier"   in out.lower()


# ===========================================================================
# 14. Integration: full pipeline on the realistic messy_df
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_produces_clean_df(self, messy_df):
        cleaned = AutoClean(messy_df).clean()
        assert cleaned["age"].isnull().sum() == 0
        assert cleaned["salary"].isnull().sum() == 0
        assert cleaned["department"].isnull().sum() == 0
        assert cleaned.duplicated().sum() == 0
        assert "empty_col" not in cleaned.columns
        assert "constant_col" in cleaned.columns   # kept
        assert 999.0 in cleaned["salary"].values   # outlier NOT removed

    def test_full_pipeline_row_count(self, messy_df):
        """11 original rows - 2 duplicate rows = 9 unique rows."""
        assert len(AutoClean(messy_df).clean()) == 9

    def test_full_pipeline_col_count(self, messy_df):
        """6 original cols - 1 empty col = 5 cols."""
        assert len(AutoClean(messy_df).clean().columns) == 5

    def test_inspect_then_clean_consistent(self, messy_df):
        cleaner = AutoClean(messy_df)
        info = cleaner.inspect()
        cleaned = cleaner.clean()
        for col in info["empty_cols"]:
            assert col not in cleaned.columns


# ===========================================================================
# 15. preview() -- read-only analysis
# ===========================================================================

class TestPreview:
    def test_preview_returns_dict(self, messy_df):
        plan = AutoClean(messy_df).preview()
        assert isinstance(plan, dict)

    def test_preview_has_required_keys(self, messy_df):
        plan = AutoClean(messy_df).preview()
        for key in (
            "col_names_to_rename", "empty_cols_to_drop", "constant_cols",
            "duplicate_rows", "impute_plan", "inconsistent_cats", "outliers",
        ):
            assert key in plan, f"Missing key: {key}"

    def test_preview_detects_empty_cols(self, messy_df):
        plan = AutoClean(messy_df).preview()
        assert "empty_col" in plan["empty_cols_to_drop"]

    def test_preview_detects_constant_cols(self, messy_df):
        plan = AutoClean(messy_df).preview()
        assert "constant_col" in plan["constant_cols"]

    def test_preview_detects_duplicates(self, messy_df):
        plan = AutoClean(messy_df).preview()
        assert plan["duplicate_rows"] == 2

    def test_preview_impute_plan_has_numerical(self, messy_df):
        plan = AutoClean(messy_df).preview()
        num_cols = [p["col"] for p in plan["impute_plan"] if p["type"] == "numerical"]
        assert "age" in num_cols
        assert "salary" in num_cols

    def test_preview_impute_plan_has_categorical(self, messy_df):
        plan = AutoClean(messy_df).preview()
        cat_cols = [p["col"] for p in plan["impute_plan"] if p["type"] == "categorical"]
        assert "department" in cat_cols

    def test_preview_impute_plan_has_fill_value(self, messy_df):
        plan = AutoClean(messy_df).preview()
        for p in plan["impute_plan"]:
            assert "fill_value" in p
            assert "strategy"   in p
            assert "missing_count" in p

    def test_preview_categorical_strategy_is_mode(self, messy_df):
        plan = AutoClean(messy_df).preview()
        cat = [p for p in plan["impute_plan"] if p["type"] == "categorical"]
        for p in cat:
            assert p["strategy"] == "mode"

    def test_preview_numerical_strategy_respected(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, np.nan]})
        plan_med = AutoClean(df, num_strategy="median").preview()
        plan_mea = AutoClean(df, num_strategy="mean").preview()
        # Both must suggest a fill value
        assert plan_med["impute_plan"][0]["fill_value"] == pytest.approx(2.0)
        assert plan_mea["impute_plan"][0]["fill_value"] == pytest.approx(2.0)

    def test_preview_detects_outliers(self, df_with_outliers):
        plan = AutoClean(df_with_outliers).preview()
        assert "score" in plan["outliers"]
        assert plan["outliers"]["score"]["count"] >= 1

    def test_preview_outlier_never_removed(self, df_with_outliers):
        """preview() must not touch the DataFrame -- outlier stays."""
        original_copy = df_with_outliers.copy()
        AutoClean(df_with_outliers).preview()
        pd.testing.assert_frame_equal(df_with_outliers, original_copy)

    def test_preview_detects_inconsistent_casing(self, messy_df):
        plan = AutoClean(messy_df).preview()
        # "department" has Engineering / engineering / ENGINEERING
        assert "department" in plan["inconsistent_cats"]

    def test_preview_inconsistent_cats_lists_variants(self, messy_df):
        plan = AutoClean(messy_df).preview()
        variants = plan["inconsistent_cats"]["department"]
        assert isinstance(variants, list)
        assert len(variants) >= 2

    def test_preview_col_rename_when_enabled(self):
        df = pd.DataFrame({"First Name": [1, 2], "Age (yrs)": [20, 30]})
        plan = AutoClean(df, clean_col_names=True).preview()
        old_names = [r[0] for r in plan["col_names_to_rename"]]
        assert "First Name" in old_names

    def test_preview_no_rename_when_disabled(self):
        df = pd.DataFrame({"First Name": [1, 2]})
        plan = AutoClean(df, clean_col_names=False).preview()
        assert plan["col_names_to_rename"] == []

    def test_preview_does_not_mutate_original(self, messy_df):
        snap = messy_df.copy()
        AutoClean(messy_df).preview()
        pd.testing.assert_frame_equal(messy_df, snap)

    def test_preview_does_not_populate_history(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.preview()
        # preview() must not log anything to history
        assert cleaner.history() == []

    def test_preview_prints_output(self, messy_df, capsys):
        AutoClean(messy_df).preview()
        out = capsys.readouterr().out
        assert "AutoClean Preview" in out
        assert "no changes applied" in out.lower()

    def test_preview_prints_impute_section(self, messy_df, capsys):
        AutoClean(messy_df).preview()
        out = capsys.readouterr().out
        assert "[IMPUTE]" in out

    def test_preview_prints_drop_section(self, messy_df, capsys):
        AutoClean(messy_df).preview()
        out = capsys.readouterr().out
        assert "[DROP]" in out

    def test_preview_prints_warn_outlier(self, df_with_outliers, capsys):
        AutoClean(df_with_outliers).preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "outlier" in out.lower()

    def test_preview_prints_warn_inconsistent(self, messy_df, capsys):
        AutoClean(messy_df).preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "inconsistent" in out.lower()

    def test_preview_clean_df_shows_no_issues(self, capsys):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        plan = AutoClean(df).preview()
        out = capsys.readouterr().out
        assert plan["duplicate_rows"] == 0
        assert plan["empty_cols_to_drop"] == []
        assert "no issues" in out.lower()


# ===========================================================================
# 16. clean(approve=False) -- preview gate
# ===========================================================================

class TestCleanApprove:
    def test_clean_approve_false_returns_none(self, simple_df):
        result = AutoClean(simple_df).clean(approve=False)
        assert result is None

    def test_clean_approve_true_returns_dataframe(self, simple_df):
        result = AutoClean(simple_df).clean(approve=True)
        assert isinstance(result, pd.DataFrame)

    def test_clean_default_is_approve_true(self, simple_df):
        """clean() with no argument behaves as approve=True."""
        result = AutoClean(simple_df).clean()
        assert isinstance(result, pd.DataFrame)

    def test_clean_approve_false_does_not_mutate(self, messy_df):
        snap = messy_df.copy()
        AutoClean(messy_df).clean(approve=False)
        pd.testing.assert_frame_equal(messy_df, snap)

    def test_clean_approve_false_does_not_log_changes(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.clean(approve=False)
        # clean(approve=False) only calls preview(), which logs nothing
        assert cleaner.history() == []

    def test_clean_approve_false_prints_preview(self, simple_df, capsys):
        AutoClean(simple_df).clean(approve=False)
        out = capsys.readouterr().out
        assert "AutoClean Preview" in out

    def test_clean_approve_false_then_true_works(self, messy_df):
        """Can preview then approve in sequence on the same cleaner."""
        cleaner = AutoClean(messy_df)
        cleaner.clean(approve=False)        # preview only
        result = cleaner.clean(approve=True)  # real clean
        assert isinstance(result, pd.DataFrame)
        assert len(cleaner.history()) > 0


# ===========================================================================
# 17. Immutability guarentees (extended)
# ===========================================================================

class TestImmutabilityExtended:
    def test_preview_does_not_alter_internal_df(self, messy_df):
        cleaner = AutoClean(messy_df)
        snap = cleaner._df.copy()
        cleaner.preview()
        pd.testing.assert_frame_equal(cleaner._df, snap)

    def test_clean_approve_false_does_not_alter_internal_df(self, messy_df):
        cleaner = AutoClean(messy_df)
        snap = cleaner._df.copy()
        cleaner.clean(approve=False)
        pd.testing.assert_frame_equal(cleaner._df, snap)

    def test_preview_does_not_alter_original_attr(self, messy_df):
        cleaner = AutoClean(messy_df)
        snap = cleaner._original.copy()
        cleaner.preview()
        pd.testing.assert_frame_equal(cleaner._original, snap)


# ===========================================================================
# 18. history() after preview + clean sequence
# ===========================================================================

class TestHistorySequence:
    def test_history_empty_after_preview_only(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.preview()
        assert cleaner.history() == []

    def test_history_populated_after_clean_following_preview(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.preview()
        cleaner.clean()
        assert len(cleaner.history()) > 0

    def test_history_has_all_steps_after_preview_then_clean(self, messy_df):
        cleaner = AutoClean(messy_df)
        cleaner.preview()
        cleaner.clean()
        steps = {e["step"] for e in cleaner.history()}
        assert "inspect"            in steps
        assert "drop_empty_cols"    in steps
        assert "drop_duplicates"    in steps
        assert "impute_numerical"   in steps
        assert "impute_categorical" in steps


# ===========================================================================
# 19. ID-like columns detection
# ===========================================================================

class TestIdLikeColumns:
    def test_id_like_detected_integer(self):
        df = pd.DataFrame({
            "user_id": list(range(1, 21)),
            "score": [50] * 20,
        })
        info = AutoClean(df).inspect()
        assert "user_id" in info["id_like_cols"]
        assert "score" not in info["id_like_cols"]

    def test_id_like_detected_string(self):
        df = pd.DataFrame({
            "uuid": [f"ID_{i}" for i in range(15)],
            "city": ["Paris"] * 15,
        })
        info = AutoClean(df).inspect()
        assert "uuid" in info["id_like_cols"]

    def test_id_like_not_removed_by_clean(self):
        df = pd.DataFrame({
            "user_id": list(range(1, 15)),
            "val": [10] * 14,
        })
        cleaned = AutoClean(df).clean()
        assert "user_id" in cleaned.columns

    def test_id_like_flagged_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "id": list(range(1, 15)),
            "val": [10] * 14,
        })
        cleaner = AutoClean(df)
        cleaner.preview()
        out_preview = capsys.readouterr().out
        assert "[FLAG]" in out_preview
        assert "id" in out_preview.lower()

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "id" in out_report.lower()


# ===========================================================================
# 20. High-cardinality categorical columns
# ===========================================================================

class TestHighCardinality:
    def test_high_cardinality_detected(self):
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(15)],
            "status": ["active"] * 15,
        })
        info = AutoClean(df).inspect()
        assert "category" in info["high_cardinality_cols"]
        assert "status" not in info["high_cardinality_cols"]

    def test_high_cardinality_not_removed_by_clean(self):
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(12)],
            "num": [1] * 12,
        })
        cleaned = AutoClean(df).clean()
        assert "category" in cleaned.columns

    def test_high_cardinality_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "code": [f"code_{i}" for i in range(15)],
        })
        cleaner = AutoClean(df)
        cleaner.preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "cardinality" in out.lower() or "code" in out.lower()

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "code" in out_report.lower()


# ===========================================================================
# 21. Inconsistent categorical values (casing and whitespace)
# ===========================================================================

class TestInconsistentCategories:
    def test_casing_and_whitespace_variants_detected(self):
        df = pd.DataFrame({
            "city": ["Mumbai", "mumbai", " Mumbai ", "Delhi", "Delhi "],
        })
        info = AutoClean(df).inspect()
        assert "city" in info["inconsistent_categories"]
        assert "city" in info["inconsistent_cats"]

    def test_suggested_normalized_value_provided(self):
        df = pd.DataFrame({
            "city": ["Mumbai", "mumbai", " Mumbai "],
        })
        info = AutoClean(df).inspect()
        clusters = info["inconsistent_categories"]["city"]
        assert len(clusters) >= 1
        assert clusters[0]["suggested"] == "Mumbai"

    def test_inconsistent_categories_not_changed_by_clean(self):
        df = pd.DataFrame({
            "city": ["Mumbai", "mumbai", " Mumbai "],
        })
        cleaned = AutoClean(df).clean()
        # Clean must NOT alter inconsistent categories
        assert "mumbai" in cleaned["city"].values
        assert " Mumbai " in cleaned["city"].values

    def test_inconsistent_categories_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "city": ["Mumbai", "mumbai"],
        })
        cleaner = AutoClean(df)
        cleaner.preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "suggested" in out.lower()
        assert "Mumbai" in out

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "suggested" in out_report.lower()


# ===========================================================================
# 22. Class imbalance (optional target)
# ===========================================================================

class TestClassImbalance:
    def test_inspect_accepts_target(self):
        df = pd.DataFrame({
            "target": ["yes"] * 9 + ["no"] * 1,
            "feature": list(range(10)),
        })
        info = AutoClean(df).inspect(target="target")
        assert info["class_imbalance"] is not None
        assert info["class_imbalance"]["target"] == "target"
        assert info["class_imbalance"]["counts"]["yes"] == 9
        assert info["class_imbalance"]["counts"]["no"] == 1
        assert info["class_imbalance"]["percentages"]["yes"] == 90.0
        assert info["class_imbalance"]["is_imbalanced"] is True

    def test_inspect_target_optional_defaults_to_none(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        info = AutoClean(df).inspect()
        assert info["class_imbalance"] is None

    def test_invalid_target_raises_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            AutoClean(df).inspect(target="non_existent")

    def test_data_not_modified_when_target_provided(self):
        df = pd.DataFrame({
            "target": ["yes"] * 8 + ["no"] * 2,
            "val": list(range(10)),
        })
        snap = df.copy()
        AutoClean(df).clean()
        pd.testing.assert_frame_equal(df, snap)

    def test_class_imbalance_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "target": ["A"] * 9 + ["B"] * 1,
        })
        cleaner = AutoClean(df, target="target")
        cleaner.preview()
        out = capsys.readouterr().out
        assert "target" in out.lower()
        assert "imbalance" in out.lower()

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "target" in out_report.lower()
        assert "imbalance" in out_report.lower()


# ===========================================================================
# 23. Possible target leakage
# ===========================================================================

class TestTargetLeakage:
    def test_direct_duplicate_leakage_detected(self):
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0, 1],
            "target_dup": [0, 1, 0, 1, 0, 1],
            "feature": [10, 20, 30, 40, 50, 60],
        })
        info = AutoClean(df).inspect(target="target")
        leaks = [l["col"] for l in info["target_leakage"]]
        assert "target_dup" in leaks
        assert "feature" not in leaks

    def test_high_correlation_leakage_detected(self):
        df = pd.DataFrame({
            "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            "leak_num": [10.001, 19.999, 30.002, 39.998, 50.001],
            "feature": [5.0, 1.0, 9.0, 2.0, 4.0],
        })
        info = AutoClean(df).inspect(target="target")
        leaks = [l["col"] for l in info["target_leakage"]]
        assert "leak_num" in leaks

    def test_target_leakage_never_removed_automatically(self):
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "target_dup": [0, 1, 0, 1, 0],
        })
        cleaned = AutoClean(df, target="target").clean()
        assert "target_dup" in cleaned.columns

    def test_target_leakage_shown_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "target_dup": [0, 1, 0, 1, 0],
        })
        cleaner = AutoClean(df, target="target")
        cleaner.preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "leakage" in out.lower()

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "leakage" in out_report.lower()


# ===========================================================================
# 24. Redundant numerical features
# ===========================================================================

class TestRedundantNumerical:
    def test_redundant_numerical_detected(self):
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "col2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # corr = 1.0
            "col3": [10.0, 5.0, 8.0, 2.0, 9.0, 1.0],
        })
        info = AutoClean(df).inspect()
        pairs = [(p["col1"], p["col2"]) for p in info["redundant_numerical"]]
        assert ("col1", "col2") in pairs or ("col2", "col1") in pairs

    def test_redundant_numerical_not_removed_by_clean(self):
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        cleaned = AutoClean(df).clean()
        assert "col1" in cleaned.columns
        assert "col2" in cleaned.columns

    def test_redundant_numerical_shown_in_preview_and_report(self, capsys):
        df = pd.DataFrame({
            "salary": [50000.0, 60000.0, 70000.0, 80000.0],
            "annual_pay": [50000.0, 60000.0, 70000.0, 80000.0],
        })
        cleaner = AutoClean(df)
        cleaner.preview()
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "redundant" in out.lower()

        cleaner.report()
        out_report = capsys.readouterr().out
        assert "redundant" in out_report.lower()


# ===========================================================================
# 25. Configuration Options & Strategies
# ===========================================================================

class TestConfigurationStrategies:
    def test_mean_numerical_strategy(self):
        df = pd.DataFrame({"num": [10.0, 20.0, 30.0, np.nan]})
        cleaner = AutoClean(df, numerical_strategy="mean")
        cleaned = cleaner.clean()
        assert cleaned["num"].iloc[3] == 20.0

    def test_median_numerical_strategy(self):
        df = pd.DataFrame({"num": [10.0, 20.0, 90.0, np.nan]})
        cleaner = AutoClean(df, numerical_strategy="median")
        cleaned = cleaner.clean()
        assert cleaned["num"].iloc[3] == 20.0

    def test_mode_categorical_strategy(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["A", "B", "A", np.nan]})
        cleaner = AutoClean(df, categorical_strategy="mode")
        cleaned = cleaner.clean()
        assert cleaned["cat"].iloc[3] == "A"

    def test_disabling_duplicate_removal(self):
        df = pd.DataFrame({"x": [1, 2, 2, 3], "y": ["a", "b", "b", "c"]})
        cleaner = AutoClean(df, remove_duplicates=False)
        cleaned = cleaner.clean()
        assert len(cleaned) == 4

    def test_enabling_duplicate_removal(self):
        df = pd.DataFrame({"x": [1, 2, 2, 3], "y": ["a", "b", "b", "c"]})
        cleaner = AutoClean(df, remove_duplicates=True)
        cleaned = cleaner.clean()
        assert len(cleaned) == 3


# ===========================================================================
# 26. Configuration Validation
# ===========================================================================

class TestConfigurationValidation:
    def test_invalid_numerical_strategy_raises_value_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid numerical_strategy"):
            AutoClean(df, numerical_strategy="invalid_strat")

    def test_invalid_categorical_strategy_raises_value_error(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        with pytest.raises(ValueError, match="Invalid categorical_strategy"):
            AutoClean(df, categorical_strategy="mean")

    def test_invalid_remove_duplicates_raises_type_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="remove_duplicates must be a boolean"):
            AutoClean(df, remove_duplicates="yes")

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            AutoClean([1, 2, 3])

    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            AutoClean(pd.DataFrame())

    def test_invalid_target_column_raises_value_error(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Target column 'bad_target' not found"):
            AutoClean(df, target="bad_target")


# ===========================================================================
# 27. Original DataFrame Unchanged
# ===========================================================================

class TestImmutabilityGuarantees:
    def test_original_dataframe_unchanged_with_custom_strategies(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 2.0],
            "b": ["x", "y", np.nan, "y"],
        })
        snap = df.copy()
        cleaner = AutoClean(df, numerical_strategy="mean", categorical_strategy="mode", remove_duplicates=False)
        cleaned = cleaner.clean()

        pd.testing.assert_frame_equal(df, snap)
        assert cleaned is not df

    def test_original_dataframe_unchanged_with_duplicate_removal(self):
        df = pd.DataFrame({"x": [1, 2, 2], "y": ["a", "b", "b"]})
        snap = df.copy()
        cleaner = AutoClean(df, remove_duplicates=True)
        cleaned = cleaner.clean()

        pd.testing.assert_frame_equal(df, snap)
        assert len(df) == 3
        assert len(cleaned) == 2


# ===========================================================================
# 28. Enhanced History
# ===========================================================================

class TestEnhancedHistoryDetails:
    def test_history_records_operation_column_affected_and_strategy(self):
        df = pd.DataFrame({
            "num": [10.0, 20.0, np.nan],
            "cat": ["A", "A", np.nan],
            "dup": [1, 1, 1],
        })
        cleaner = AutoClean(df, numerical_strategy="mean", categorical_strategy="mode", remove_duplicates=True)
        cleaner.clean()

        history = cleaner.history()
        assert len(history) > 0

        # Check required fields on every entry
        for entry in history:
            assert "operation" in entry
            assert "column" in entry
            assert "affected_count" in entry
            assert "strategy" in entry

        # Check numerical imputation entry
        num_entries = [e for e in history if e["operation"] == "impute_numerical"]
        assert len(num_entries) == 1
        assert num_entries[0]["column"] == "num"
        assert num_entries[0]["affected_count"] == 1
        assert num_entries[0]["strategy"] == "mean"

        # Check categorical imputation entry
        cat_entries = [e for e in history if e["operation"] == "impute_categorical"]
        assert len(cat_entries) == 1
        assert cat_entries[0]["column"] == "cat"
        assert cat_entries[0]["affected_count"] == 1
        assert cat_entries[0]["strategy"] == "mode"


# ===========================================================================
# 29. reset()
# ===========================================================================

class TestCleanerResetMethod:
    def test_reset_clears_history(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan],
            "b": ["x", "x", "x"],
        })
        cleaner = AutoClean(df)
        cleaner.clean()
        assert len(cleaner.history()) > 0

        cleaner.reset()
        assert cleaner.history() == []

    def test_reset_allows_analyzing_original_again(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, np.nan],
            "b": ["x", "x", "x"],
        })
        cleaner = AutoClean(df)
        cleaned1 = cleaner.clean()
        assert cleaned1["a"].isnull().sum() == 0

        cleaner.reset()
        # After reset, inspect reflects original missing value again
        info = cleaner.inspect()
        assert info["missing"]["a"] == 1

        cleaned2 = cleaner.clean()
        assert cleaned2["a"].isnull().sum() == 0
        pd.testing.assert_frame_equal(cleaned1, cleaned2)

    def test_reset_returns_self_for_chaining(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        cleaner = AutoClean(df)
        cleaner.clean()
        res = cleaner.reset()
        assert res is cleaner


# ===========================================================================
# 30. Edge-Case Dataset 1: Small dataset (2-3 rows)
# ===========================================================================

class TestEdgeCaseSmallDataset:
    def test_small_dataset_pipeline(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan],
            "b": ["x", "x"],
        })
        snap = df.copy()
        cleaner = AutoClean(df, numerical_strategy="mean", remove_duplicates=False)

        # inspect() never modifies input
        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["missing"]["a"] == 1

        # preview() never modifies input
        plan = cleaner.preview()
        pd.testing.assert_frame_equal(df, snap)

        # clean() never modifies input and returns valid DataFrame
        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned["a"].isnull().sum() == 0
        assert cleaned["a"].iloc[1] == 1.0


# ===========================================================================
# 31. Edge-Case Dataset 2: All values missing in a column
# ===========================================================================

class TestEdgeCaseAllMissingColumn:
    def test_all_missing_column_dropped(self):
        df = pd.DataFrame({
            "valid": [1, 2, 3],
            "all_nan": [np.nan, np.nan, np.nan],
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert "all_nan" in info["empty_cols"]

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert "all_nan" not in cleaned.columns
        assert "valid" in cleaned.columns


# ===========================================================================
# 32. Edge-Case Dataset 3: No missing values
# ===========================================================================

class TestEdgeCaseNoMissingValues:
    def test_no_missing_values(self, capsys):
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["missing"] == {}

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert len(cleaner.history()) == 1  # only inspect logged
        assert cleaned["num"].tolist() == [1.0, 2.0, 3.0]

        # report() works when there are no changes
        cleaner.report()
        out = capsys.readouterr().out
        assert "no changes were necessary" in out.lower() or "missing values : none" in out.lower()


# ===========================================================================
# 33. Edge-Case Dataset 4: No duplicates
# ===========================================================================

class TestEdgeCaseNoDuplicates:
    def test_no_duplicates_handling(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "y", "z"],
        })
        snap = df.copy()
        cleaner = AutoClean(df, remove_duplicates=True)

        info = cleaner.inspect()
        assert info["duplicates"] == 0

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert len(cleaned) == 4
        # duplicate removal step not needed / 0 removed
        dup_steps = [e for e in cleaner.history() if e["operation"] == "drop_duplicates"]
        assert len(dup_steps) == 0


# ===========================================================================
# 34. Edge-Case Dataset 5: Only numerical columns
# ===========================================================================

class TestEdgeCaseOnlyNumericalColumns:
    def test_only_numerical_columns(self):
        df = pd.DataFrame({
            "x": [10.0, 20.0, np.nan, 40.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "z": [100.0, 200.0, 300.0, 400.0],
        })
        snap = df.copy()
        cleaner = AutoClean(df, numerical_strategy="median")

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert all(t == "numerical" for t in info["col_types"].values())

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert cleaned["x"].isnull().sum() == 0
        assert cleaned["x"].iloc[2] == 20.0


# ===========================================================================
# 35. Edge-Case Dataset 6: Only categorical columns
# ===========================================================================

class TestEdgeCaseOnlyCategoricalColumns:
    def test_only_categorical_columns(self):
        df = pd.DataFrame({
            "c1": ["apple", "banana", "apple", None],
            "c2": ["red", "yellow", "red", "green"],
        })
        snap = df.copy()
        cleaner = AutoClean(df, categorical_strategy="mode")

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert all(t == "categorical" for t in info["col_types"].values())

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert cleaned["c1"].isnull().sum() == 0
        assert cleaned["c1"].iloc[2] == "apple"


# ===========================================================================
# 36. Edge-Case Dataset 7: Containing datetime columns
# ===========================================================================

class TestEdgeCaseDatetimeColumns:
    def test_datetime_columns_not_imputed_or_corrupted(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2021-01-01", "2021-06-15", None, "2021-12-31"]),
            "val": [10.0, 20.0, np.nan, 40.0],
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["col_types"]["dt"] == "datetime"

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["dt"])
        assert cleaned["val"].isnull().sum() == 0
        # datetime missing value preserved without domain assumption
        assert cleaned["dt"].isnull().sum() == 1


# ===========================================================================
# 37. Edge-Case Dataset 8: Mixed / invalid values
# ===========================================================================

class TestEdgeCaseMixedInvalidValues:
    def test_mixed_object_values_handled_gracefully(self):
        df = pd.DataFrame({
            "mixed": [1, "two", 3.0, None, "two"],
            "score": [10.0, 20.0, np.nan, 40.0, 50.0],
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["col_types"]["mixed"] == "categorical"

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert cleaned["mixed"].isnull().sum() == 0
        assert cleaned["score"].isnull().sum() == 0


# ===========================================================================
# 38. Edge-Case Dataset 9: Very large numerical outliers
# ===========================================================================

class TestEdgeCaseVeryLargeOutliers:
    def test_extreme_outliers_flagged_but_never_deleted(self):
        df = pd.DataFrame({
            "val": [10.0, 11.0, 10.5, 9.5, 10.2, 1e15, -1e15],
            "id": list(range(7)),
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["outliers"]["val"]["count"] >= 1

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        # Outliers must NOT be deleted
        assert len(cleaned) == 7
        assert 1e15 in cleaned["val"].values
        assert -1e15 in cleaned["val"].values


# ===========================================================================
# 39. Edge-Case Dataset 10: Single-row DataFrame
# ===========================================================================

class TestEdgeCaseSingleRowDataFrame:
    def test_single_row_dataframe_all_methods(self, capsys):
        df = pd.DataFrame({"x": [42.0], "name": ["Antigravity"]})
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert info["shape"] == (1, 2)
        assert info["duplicates"] == 0

        plan = cleaner.preview()
        pd.testing.assert_frame_equal(df, snap)
        assert plan["duplicate_rows"] == 0

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) == 1
        assert cleaned["x"].iloc[0] == 42.0

        cleaner.report()
        out = capsys.readouterr().out
        assert "AutoClean Report" in out


# ===========================================================================
# 40. Edge-Case Dataset 11: Empty DataFrame
# ===========================================================================

class TestEdgeCaseEmptyDataFrame:
    def test_empty_dataframe_raises_clear_error(self):
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            AutoClean(pd.DataFrame())

    def test_dataframe_with_columns_but_zero_rows_raises_error(self):
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            AutoClean(pd.DataFrame({"a": [], "b": []}))


# ===========================================================================
# 41. Constant Column & Immutability Verification
# ===========================================================================

class TestEdgeCaseConstantColumnsAndImmutability:
    def test_constant_column_detected_and_kept(self):
        df = pd.DataFrame({
            "const_num": [5.0, 5.0, 5.0, 5.0],
            "const_str": ["YES", "YES", "YES", "YES"],
            "normal": [1, 2, 3, 4],
        })
        snap = df.copy()
        cleaner = AutoClean(df)

        info = cleaner.inspect()
        pd.testing.assert_frame_equal(df, snap)
        assert "const_num" in info["constant_cols"]
        assert "const_str" in info["constant_cols"]

        cleaned = cleaner.clean()
        pd.testing.assert_frame_equal(df, snap)
        # Constant columns are flagged only, NEVER automatically dropped
        assert "const_num" in cleaned.columns
        assert "const_str" in cleaned.columns
        assert "normal" in cleaned.columns

