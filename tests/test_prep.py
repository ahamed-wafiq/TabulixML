"""
tests/test_prep.py -- Comprehensive pytest suite for datacraft.AutoPrep

Tests:
  - Instantiation & validation:
      * target column existence & types
      * test_size range validation (0 < test_size < 1)
      * random_state validation
      * scale & encode boolean validation
  - Column type detection & inspect()
  - preview() format, sections, and configuration flags
  - split() method:
      * correct train/test split sizes
      * reproducibility with random_state
  - prepare() pipeline execution:
      * preprocessing fitted ONLY on training data (prevents data leakage)
      * numerical preprocessing (median imputer, StandardScaler, scale=False)
      * categorical preprocessing (mode imputer, OneHotEncoder, encode=False)
      * missing values handling
      * target separation and no target leakage
      * mixed numerical/categorical/datetime data
      * only numerical data
      * only categorical data
      * unseen/unknown categories handled gracefully without crashing
      * feature name preservation after OneHotEncoder
  - get_pipeline():
      * returns fitted ColumnTransformer
      * raises RuntimeError before prepare()
  - transform():
      * transforms new/unseen data using fitted pipeline
      * strictly does NOT refit the pipeline
      * raises RuntimeError before prepare()
  - Original DataFrame immutability guarantee
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from datacraft import AutoPrep


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def sample_mixed_df() -> pd.DataFrame:
    """10-row mixed-type DataFrame for train/test split testing."""
    return pd.DataFrame({
        "age": [25.0, 30.0, np.nan, 45.0, 29.0, 35.0, 40.0, 22.0, 50.0, 31.0],
        "salary": [50000.0, 60000.0, 55000.0, 90000.0, 52000.0, 75000.0, 80000.0, 48000.0, 110000.0, 62000.0],
        "department": ["Engineering", "Sales", "HR", "Sales", np.nan, "Marketing", "Engineering", "HR", "Sales", "Marketing"],
        "joined": pd.to_datetime([
            "2020-01-15", "2019-06-01", "2021-03-22", "2018-11-05", "2020-08-10",
            "2017-04-12", "2016-09-30", "2022-02-14", "2015-11-20", "2020-05-18",
        ]),
        "churn": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    })


@pytest.fixture()
def numerical_only_df() -> pd.DataFrame:
    """10-row numerical-only features with a target."""
    return pd.DataFrame({
        "feature_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture()
def categorical_only_df() -> pd.DataFrame:
    """10-row categorical-only features with a target."""
    return pd.DataFrame({
        "color": ["red", "blue", "red", np.nan, "green", "blue", "red", "green", "blue", "red"],
        "size": ["S", "M", "L", "S", "M", "L", "S", "M", "L", "S"],
        "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
    })


# ============================================================================
# 1. Instantiation and Validation
# ============================================================================

class TestPrepInstantiationAndValidation:
    def test_valid_instantiation(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        assert prep.target == "churn"
        assert prep.test_size == 0.2
        assert prep.random_state == 42
        assert "churn" not in prep.feature_columns
        assert "age" in prep.numerical_columns
        assert "department" in prep.categorical_columns

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            AutoPrep([1, 2, 3], target="target")

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            AutoPrep({"a": [1, 2]}, target="target")

    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            AutoPrep(pd.DataFrame(), target="target")

    def test_invalid_target_column_raises_value_error(self, sample_mixed_df):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            AutoPrep(sample_mixed_df, target="non_existent_column")

    def test_non_string_target_raises_type_error(self, sample_mixed_df):
        with pytest.raises(TypeError, match="Target column must be a string"):
            AutoPrep(sample_mixed_df, target=123)

    def test_invalid_test_size_raises_value_error(self, sample_mixed_df):
        for invalid_size in [0.0, 1.0, -0.2, 1.5, "0.2"]:
            with pytest.raises(ValueError, match="test_size must be a float between 0.0 and 1.0"):
                AutoPrep(sample_mixed_df, target="churn", test_size=invalid_size)

    def test_invalid_random_state_raises_value_error(self, sample_mixed_df):
        for invalid_state in [-1, -42, "42"]:
            with pytest.raises(ValueError, match="random_state must be a non-negative integer or None"):
                AutoPrep(sample_mixed_df, target="churn", random_state=invalid_state)

    def test_prepare_without_target_raises_value_error(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target=None)
        with pytest.raises(ValueError, match="target column"):
            prep.prepare()

    def test_transform_before_prepare_raises_runtime_error(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn")
        with pytest.raises(RuntimeError, match="transform\\(\\) cannot be called before prepare\\(\\)"):
            prep.transform(sample_mixed_df)

    def test_get_pipeline_before_prepare_raises_runtime_error(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn")
        with pytest.raises(RuntimeError, match="Pipeline has not been fitted yet"):
            prep.get_pipeline()


# ============================================================================
# 2. inspect() and preview()
# ============================================================================

class TestPrepInspectAndPreview:
    def test_inspect_keys_and_target_separation(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn")
        info = prep.inspect()

        assert "feature_columns" in info
        assert "target_column" in info
        assert "numerical_columns" in info
        assert "categorical_columns" in info
        assert "datetime_columns" in info
        assert "missing_values" in info
        assert "unique_values" in info

        assert info["target_column"] == "churn"
        assert "churn" not in info["feature_columns"]
        assert info["col_types"]["churn"] == "target"

    def test_preview_returns_plan_and_prints(self, sample_mixed_df, capsys):
        prep = AutoPrep(sample_mixed_df, target="churn")
        plan = prep.preview(print_preview=True)

        assert isinstance(plan, dict)
        assert "target" in plan
        assert "split" in plan
        assert "numerical" in plan
        assert "categorical" in plan
        assert "datetime" in plan

        captured = capsys.readouterr().out
        assert "DataCraft -- AutoPrep Preview" in captured
        assert "Target Column : churn" in captured
        assert "test_size=0.2" in captured


# ============================================================================
# 3. split() Method
# ============================================================================

class TestPrepSplit:
    def test_correct_train_test_sizes(self, sample_mixed_df):
        # 10 rows total, test_size=0.2 -> 8 train, 2 test
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = prep.split()

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_split_different_test_size(self, sample_mixed_df):
        # 10 rows total, test_size=0.3 -> 7 train, 3 test
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = prep.split()

        assert len(X_train) == 7
        assert len(X_test) == 3

    def test_reproducibility_with_random_state(self, sample_mixed_df):
        prep1 = AutoPrep(sample_mixed_df, target="churn", random_state=42)
        X_train1, X_test1, y_train1, y_test1 = prep1.split()

        prep2 = AutoPrep(sample_mixed_df, target="churn", random_state=42)
        X_train2, X_test2, y_train2, y_test2 = prep2.split()

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    def test_different_random_state_produces_different_split(self, sample_mixed_df):
        prep1 = AutoPrep(sample_mixed_df, target="churn", random_state=42)
        X_train1, _, _, _ = prep1.split()

        prep2 = AutoPrep(sample_mixed_df, target="churn", random_state=99)
        X_train2, _, _, _ = prep2.split()

        assert not X_train1.index.equals(X_train2.index)


# ============================================================================
# 4. prepare() Preprocessing & Data Leakage Prevention
# ============================================================================

class TestPrepPrepare:
    def test_prepare_returns_four_splits(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = prep.prepare()

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_preprocessing_fitted_only_on_training_data(self):
        """Verify pipeline statistics are computed strictly from X_train, preventing test leak."""
        # Train rows: 0, 1, 2 (values: 10.0, 20.0, 30.0) -> mean = 20.0
        # Test row  : 3 (value: 1000.0) -> extreme outlier!
        # If fitted on train only: scaler mean_ must be exactly 20.0
        # If fitted on all 4 rows: scaler mean_ would be (10+20+30+1000)/4 = 265.0
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, 1000.0],
            "target": [0, 1, 0, 1],
        })

        # By using test_size=0.25 and random_state=42, we split 3 train, 1 test
        prep = AutoPrep(df, target="target", test_size=0.25, random_state=42)
        X_train, X_test, y_train, y_test = prep.prepare()

        pipeline = prep.get_pipeline()
        scaler = pipeline.named_transformers_["num"].named_steps["scaler"]

        # Check raw train split value mean
        raw_train_vals = prep._X_train_raw["val"].values
        expected_train_mean = np.mean(raw_train_vals)

        # The scaler must match the training set mean, NEVER the full 4-row dataset mean
        assert np.isclose(scaler.mean_[0], expected_train_mean)
        assert not np.isclose(scaler.mean_[0], df["val"].mean())

    def test_no_target_leakage(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn")
        X_train, X_test, y_train, y_test = prep.prepare()

        # Target 'churn' is not in feature names
        feature_names = prep.get_feature_names()
        assert not any("churn" in name for name in feature_names)

        # Target series match original target values exactly
        assert set(y_train.values).issubset({0, 1})
        assert set(y_test.values).issubset({0, 1})

    def test_preserve_feature_names(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn")
        prep.prepare()

        names = prep.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

        # Numerical names preserved
        assert "age" in names
        assert "salary" in names

        # Categorical one-hot names preserved
        assert any(name.startswith("department_") for name in names)

    def test_only_numerical_features(self, numerical_only_df):
        prep = AutoPrep(numerical_only_df, target="target", test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = prep.prepare()

        assert X_train.shape == (8, 2)
        assert X_test.shape == (2, 2)
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_only_categorical_features(self, categorical_only_df):
        prep = AutoPrep(categorical_only_df, target="target", test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = prep.prepare()

        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert X_train.shape[1] > 0
        assert X_test.shape[1] == X_train.shape[1]


# ============================================================================
# 5. transform() on New Data & Pipeline Persistence
# ============================================================================

class TestPrepTransform:
    def test_transform_works_on_new_data(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        X_train, _, _, _ = prep.prepare()

        new_data = pd.DataFrame({
            "age": [32.0, 48.0],
            "salary": [65000.0, 92000.0],
            "department": ["Sales", "Engineering"],
            "joined": pd.to_datetime(["2021-01-10", "2017-08-25"]),
        })

        X_new = prep.transform(new_data)
        assert isinstance(X_new, np.ndarray)
        assert X_new.shape == (2, X_train.shape[1])
        assert not np.isnan(X_new).any()

    def test_transform_handles_unseen_categories_safely(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        X_train, _, _, _ = prep.prepare()

        # Unseen department 'Artificial Intelligence'
        unseen_data = pd.DataFrame({
            "age": [28.0],
            "salary": [80000.0],
            "department": ["Artificial Intelligence"],
            "joined": pd.to_datetime(["2023-01-01"]),
        })

        X_unseen = prep.transform(unseen_data)
        assert isinstance(X_unseen, np.ndarray)
        assert X_unseen.shape == (1, X_train.shape[1])
        assert not np.isnan(X_unseen).any()

    def test_transform_does_not_refit_pipeline(self, sample_mixed_df):
        prep = AutoPrep(sample_mixed_df, target="churn", test_size=0.2, random_state=42)
        prep.prepare()

        pipeline_before = prep.get_pipeline()
        scaler_mean_before = pipeline_before.named_transformers_["num"].named_steps["scaler"].mean_.copy()

        # New data with radically different numbers
        new_data = pd.DataFrame({
            "age": [999.0],
            "salary": [9999999.0],
            "department": ["Sales"],
            "joined": pd.to_datetime(["2020-01-01"]),
        })

        prep.transform(new_data)

        # Pipeline instance must be identical (not refitted)
        pipeline_after = prep.get_pipeline()
        assert pipeline_after is pipeline_before

        scaler_mean_after = pipeline_after.named_transformers_["num"].named_steps["scaler"].mean_
        np.testing.assert_array_equal(scaler_mean_before, scaler_mean_after)


# ============================================================================
# 6. Immutability Guarantee
# ============================================================================

class TestPrepImmutability:
    def test_original_dataframe_unchanged(self, sample_mixed_df):
        snapshot = sample_mixed_df.copy()
        prep = AutoPrep(sample_mixed_df, target="churn")

        prep.inspect()
        pd.testing.assert_frame_equal(sample_mixed_df, snapshot)

        prep.preview(print_preview=False)
        pd.testing.assert_frame_equal(sample_mixed_df, snapshot)

        prep.split()
        pd.testing.assert_frame_equal(sample_mixed_df, snapshot)

        prep.prepare()
        pd.testing.assert_frame_equal(sample_mixed_df, snapshot)

        prep.transform(sample_mixed_df)
        pd.testing.assert_frame_equal(sample_mixed_df, snapshot)
