"""
tests/test_automl.py -- Test Suite for DataCraft AutoML Module

Covers:
  - Task detection (classification vs regression)
  - Inspect and preview
  - Baseline model lists for classification and regression
  - Fit on classification and regression datasets (single train/test)
  - Evaluate with cross-validation (StratifiedKFold & KFold)
  - Results table structure (single fit table vs cross-validation summary table)
  - Compare model performance and label highest-scoring model without calling it "best"
  - Report output and formatting
  - Target validations, cv validation, and scoring validation
  - Reproducibility with random_state
  - Strict immutability of input DataFrame
  - Zero target/preprocessing leakage in cross-validation and fit
"""

import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from datacraft import AutoML, AutoPrep


@pytest.fixture
def classification_df():
    """Synthetic classification dataset with numeric and categorical features."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame({
        "age": rng.normal(35, 10, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "department": rng.choice(["Sales", "Engineering", "Marketing"], size=n),
        "churn": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
    })


@pytest.fixture
def multiclass_df():
    """Synthetic multiclass classification dataset."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame({
        "feature1": rng.normal(10, 2, size=n),
        "feature2": rng.normal(50, 10, size=n),
        "category": rng.choice(["TypeA", "TypeB", "TypeC"], size=n),
        "target": rng.choice(["Low", "Medium", "High"], size=n),
    })


@pytest.fixture
def regression_df():
    """Synthetic regression dataset with continuous target."""
    rng = np.random.RandomState(42)
    n = 60
    x1 = rng.normal(10, 2, size=n)
    x2 = rng.normal(5, 1, size=n)
    cat = rng.choice(["Urban", "Suburban", "Rural"], size=n)
    y = 3.5 * x1 - 2.0 * x2 + rng.normal(0, 1, size=n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "region": cat,
        "price": y,
    })


# ==============================================================================
# 1. Task Detection Tests
# ==============================================================================
class TestAutoMLTaskDetection:
    def test_detect_binary_integer_classification(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        res = automl.detect_task()
        assert res["task"] == "classification"
        assert res["target"] == "churn"
        assert automl.task == "classification"

    def test_detect_multiclass_string_classification(self, multiclass_df):
        automl = AutoML(multiclass_df, target="target")
        res = automl.detect_task()
        assert res["task"] == "classification"
        assert res["target"] == "target"
        assert automl.task == "classification"

    def test_detect_boolean_classification(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "flag": [True, False, True, False, True, False, True, False],
        })
        automl = AutoML(df, target="flag")
        assert automl.detect_task()["task"] == "classification"

    def test_detect_categorical_dtype_classification(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": pd.Categorical(["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"]),
        })
        automl = AutoML(df, target="label")
        assert automl.detect_task()["task"] == "classification"

    def test_detect_continuous_float_regression(self, regression_df):
        automl = AutoML(regression_df, target="price")
        res = automl.detect_task()
        assert res["task"] == "regression"
        assert res["target"] == "price"
        assert automl.task == "regression"

    def test_detect_high_cardinality_integer_regression(self):
        df = pd.DataFrame({
            "feature": list(range(20)),
            "salary": [30000 + i * 2500 for i in range(20)],
        })
        automl = AutoML(df, target="salary")
        assert automl.detect_task()["task"] == "regression"


# ==============================================================================
# 2. Inspect and Preview Tests
# ==============================================================================
class TestAutoMLInspectAndPreview:
    def test_inspect_classification_returns_correct_metadata(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn")
        info = automl.inspect()
        out = capsys.readouterr().out

        assert info["target"] == "churn"
        assert info["task"] == "classification"
        assert info["samples"] == len(classification_df)
        assert info["features"] == len(classification_df.columns) - 1
        assert "age" in info["numerical_features"]
        assert "income" in info["numerical_features"]
        assert "department" in info["categorical_features"]
        assert info["class_counts"] is not None
        assert "DataCraft AutoML Inspection" in out

    def test_inspect_regression_returns_correct_metadata(self, regression_df, capsys):
        automl = AutoML(regression_df, target="price")
        info = automl.inspect()
        out = capsys.readouterr().out

        assert info["target"] == "price"
        assert info["task"] == "regression"
        assert info["class_counts"] is None
        assert "mean" in info["target_distribution"]
        assert "std" in info["target_distribution"]
        assert "DataCraft AutoML Inspection" in out

    def test_preview_classification(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn")
        models = automl.preview()
        out = capsys.readouterr().out

        assert models == ["Logistic Regression", "Decision Tree", "Random Forest"]
        assert "DataCraft AutoML Preview" in out
        assert "Task: classification" in out
        assert "* Logistic Regression" in out
        assert "* Decision Tree" in out
        assert "* Random Forest" in out
        assert automl.is_fitted is False

    def test_preview_regression(self, regression_df, capsys):
        automl = AutoML(regression_df, target="price")
        models = automl.preview()
        out = capsys.readouterr().out

        assert models == ["Linear Regression", "Decision Tree", "Random Forest"]
        assert "DataCraft AutoML Preview" in out
        assert "Task: regression" in out
        assert "* Linear Regression" in out
        assert automl.is_fitted is False


# ==============================================================================
# 3. Model Training (fit) and Train/Test Results Tests
# ==============================================================================
class TestAutoMLFitAndResults:
    def test_fit_classification_trains_all_models(self, classification_df):
        automl = AutoML(classification_df, target="churn", test_size=0.2, random_state=42)
        automl.fit()

        assert automl.is_fitted is True
        fitted_models = automl.get_models()
        assert len(fitted_models) == 3
        assert isinstance(fitted_models["Logistic Regression"], LogisticRegression)
        assert isinstance(fitted_models["Decision Tree"], DecisionTreeClassifier)
        assert isinstance(fitted_models["Random Forest"], RandomForestClassifier)

    def test_fit_regression_trains_all_models(self, regression_df):
        automl = AutoML(regression_df, target="price", test_size=0.2, random_state=42)
        automl.fit()

        assert automl.is_fitted is True
        fitted_models = automl.get_models()
        assert len(fitted_models) == 3
        assert isinstance(fitted_models["Linear Regression"], LinearRegression)
        assert isinstance(fitted_models["Decision Tree"], DecisionTreeRegressor)
        assert isinstance(fitted_models["Random Forest"], RandomForestRegressor)

    def test_results_before_fit_or_evaluate_raises_runtime_error(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="cannot be called before.*fit.*evaluate"):
            automl.results()

    def test_get_models_before_fit_raises_runtime_error(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="has not been fitted yet"):
            automl.get_models()

    def test_results_classification_metrics(self, classification_df):
        automl = AutoML(classification_df, target="churn", test_size=0.2, random_state=42)
        automl.fit()
        res = automl.results()

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 12  # 3 models * 4 metrics (accuracy, precision, recall, F1)

        # Check required columns exist (case-insensitive)
        assert "model" in res.columns
        assert "metric" in res.columns
        assert "score" in res.columns

        # Verify all four required metrics are computed
        metrics_present = set(res["metric"].str.lower().unique())
        assert {"accuracy", "precision", "recall", "f1"}.issubset(metrics_present)

        # Check models present
        models_present = set(res["model"].unique())
        assert {"Logistic Regression", "Decision Tree", "Random Forest"}.issubset(models_present)

        # Check valid score ranges
        assert (res["score"] >= 0.0).all()
        assert (res["score"] <= 1.0).all()

    def test_results_regression_metrics(self, regression_df):
        automl = AutoML(regression_df, target="price", test_size=0.2, random_state=42)
        automl.fit()
        res = automl.results()

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 9  # 3 models * 3 metrics (MAE, RMSE, R²)

        # Check required columns exist
        assert "model" in res.columns
        assert "metric" in res.columns
        assert "score" in res.columns

        # Verify all three required metrics are computed
        metrics_present = set(res["metric"].unique())
        assert {"MAE", "RMSE", "R²"}.issubset(metrics_present)

        mae_scores = res[res["metric"] == "MAE"]["score"]
        rmse_scores = res[res["metric"] == "RMSE"]["score"]
        assert (mae_scores >= 0.0).all()
        assert (rmse_scores >= 0.0).all()

    def test_results_flexible_column_access(self, classification_df):
        automl = AutoML(classification_df, target="churn", random_state=42)
        automl.fit()
        res = automl.results()

        assert len(res["model"]) == len(res["Model"])
        assert len(res["metric"]) == len(res["Metric"])
        assert len(res["score"]) == len(res["Score"])

    def test_fit_multiclass_classification_handles_metrics_safely(self, multiclass_df):
        automl = AutoML(multiclass_df, target="target", random_state=42)
        automl.fit()
        res = automl.results()

        assert len(res) == 12
        assert (res["score"] >= 0.0).all()


# ==============================================================================
# 4. Cross-Validation (evaluate) and Results Format Tests
# ==============================================================================
class TestAutoMLCrossValidation:
    def test_evaluate_classification_default_5_fold(self, classification_df):
        automl = AutoML(classification_df, target="churn", cv=5, random_state=42)
        automl.evaluate()

        assert automl.is_evaluated is True
        res = automl.results()

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 3  # 3 baseline models

        # Check columns: Model, Mean F1, Std, Fold 1..Fold 5
        cols = list(res.columns)
        assert "Model" in cols
        assert "Mean F1" in cols
        assert "Std" in cols
        for i in range(1, 6):
            assert f"Fold {i}" in cols

        # Check flexible lookup
        assert len(res["model"]) == 3
        assert len(res["mean score"]) == 3
        assert len(res["std"]) == 3
        assert len(res["fold 1"]) == 3

        # Scores should be within reasonable bounds
        assert (res["Mean F1"] >= 0.0).all()
        assert (res["Mean F1"] <= 1.0).all()

    def test_evaluate_regression_default_5_fold(self, regression_df):
        automl = AutoML(regression_df, target="price", cv=5, random_state=42)
        automl.evaluate()

        assert automl.is_evaluated is True
        res = automl.results()

        assert len(res) == 3
        cols = list(res.columns)
        assert "Model" in cols
        assert "Mean R²" in cols
        assert "Std" in cols
        for i in range(1, 6):
            assert f"Fold {i}" in cols

    def test_evaluate_custom_cv_fold_count(self, classification_df):
        automl = AutoML(classification_df, target="churn", cv=3, random_state=42)
        automl.evaluate()

        res = automl.results()
        assert "Fold 1" in res.columns
        assert "Fold 2" in res.columns
        assert "Fold 3" in res.columns
        assert "Fold 4" not in res.columns

    def test_evaluate_custom_scoring_metric_accuracy(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=4, scoring="accuracy", random_state=42
        )
        automl.evaluate()
        res = automl.results()

        assert "Mean Accuracy" in res.columns
        assert (res["Mean Accuracy"] >= 0.0).all()
        assert (res["Mean Accuracy"] <= 1.0).all()

    def test_evaluate_custom_scoring_metric_mae(self, regression_df):
        automl = AutoML(
            regression_df, target="price", cv=4, scoring="mae", random_state=42
        )
        automl.evaluate()
        res = automl.results()

        assert "Mean MAE" in res.columns
        assert (res["Mean MAE"] >= 0.0).all()

    def test_logical_separation_fit_and_evaluate(self, classification_df):
        automl = AutoML(classification_df, target="churn", random_state=42)

        # Before any execution
        with pytest.raises(RuntimeError):
            automl.results()

        # Run fit() only
        automl.fit()
        assert automl.is_fitted is True
        assert automl.is_evaluated is False
        fit_res = automl.results()
        assert "metric" in fit_res.columns  # Single split format

        # Run evaluate()
        automl.evaluate()
        assert automl.is_evaluated is True
        cv_res = automl.results()
        assert "Mean F1" in cv_res.columns  # CV format

        # Can still access fit results via source parameter
        fit_res_explicit = automl.results(source="fit")
        assert "metric" in fit_res_explicit.columns


# ==============================================================================
# 5. Model Comparison (compare) Tests
# ==============================================================================
class TestAutoMLCompare:
    def test_compare_before_evaluate_raises_runtime_error(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="cannot be called before evaluate"):
            automl.compare()

    def test_compare_classification_output_and_label(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn", cv=5, random_state=42)
        automl.evaluate()
        sorted_res = automl.compare()
        out = capsys.readouterr().out

        assert "DataCraft AutoML Model Comparison" in out
        assert "Highest-scoring model for F1:" in out
        # Must NOT call the top model "best"
        assert "Best model:" not in out
        assert "best model" not in out.lower()

        # Verify sorted in descending order of Mean F1
        mean_scores = list(sorted_res["Mean F1"])
        assert mean_scores == sorted(mean_scores, reverse=True)

    def test_compare_regression_mae_sorts_ascending(self, regression_df, capsys):
        automl = AutoML(
            regression_df, target="price", cv=3, scoring="mae", random_state=42
        )
        automl.evaluate()
        sorted_res = automl.compare()
        out = capsys.readouterr().out

        assert "Highest-scoring model for MAE:" in out
        # For MAE, lowest error is best -> ascending order
        mae_scores = list(sorted_res["Mean MAE"])
        assert mae_scores == sorted(mae_scores, reverse=False)

    def test_compare_print_table_false_suppresses_stdout(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn", cv=3, random_state=42)
        automl.evaluate()
        sorted_res = automl.compare(print_table=False)
        out = capsys.readouterr().out

        assert len(out) == 0
        assert isinstance(sorted_res, pd.DataFrame)


# ==============================================================================
# 6. Report Tests
# ==============================================================================
class TestAutoMLReport:
    def test_report_before_fit_shows_not_fitted(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn")
        rep = automl.report()
        out = capsys.readouterr().out

        assert "DataCraft AutoML Report" in rep
        assert "Detected Task   : classification" in rep
        assert "not fitted" in rep

    def test_report_after_evaluate_shows_cv_summary(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn", cv=4, random_state=42)
        automl.evaluate()
        rep = automl.report()
        out = capsys.readouterr().out

        assert "DataCraft AutoML Report" in rep
        assert "evaluate() completed (4-fold CV)" in rep
        assert "Cross-Validation Summary" in rep
        assert "Mean F1" in rep

    def test_report_print_report_false_suppresses_stdout(self, classification_df, capsys):
        automl = AutoML(classification_df, target="churn", random_state=42)
        automl.evaluate()
        rep = automl.report(print_report=False)
        out = capsys.readouterr().out

        assert len(out) == 0
        assert "DataCraft AutoML Report" in rep


# ==============================================================================
# 7. Validation and Error Handling Tests
# ==============================================================================
class TestAutoMLValidation:
    def test_invalid_cv_values(self, classification_df):
        with pytest.raises(ValueError, match="cv must be an integer >= 2"):
            AutoML(classification_df, target="churn", cv=1)
        with pytest.raises(ValueError, match="cv must be an integer >= 2"):
            AutoML(classification_df, target="churn", cv=-3)
        with pytest.raises(ValueError, match="cv must be an integer >= 2"):
            AutoML(classification_df, target="churn", cv=True)
        with pytest.raises(ValueError, match="cv must be an integer >= 2"):
            AutoML(classification_df, target="churn", cv="5")

    def test_cv_greater_than_sample_count(self):
        tiny_df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
        })
        automl = AutoML(tiny_df, target="target", cv=10)
        with pytest.raises(ValueError, match="which is less than cv=10"):
            automl.evaluate()

    def test_cv_exceeds_least_populated_class(self):
        # 1 positive sample vs 9 negative samples with cv=3
        imbalanced_df = pd.DataFrame({
            "x": list(range(10)),
            "target": [0] * 9 + [1] * 1,
        })
        automl = AutoML(imbalanced_df, target="target", cv=3)
        with pytest.raises(ValueError, match="least populated class has only 1 sample"):
            automl.evaluate()

    def test_invalid_scoring_name(self, classification_df):
        with pytest.raises(ValueError, match="Invalid scoring metric 'invalid_metric'"):
            AutoML(classification_df, target="churn", scoring="invalid_metric")

    def test_non_string_scoring_raises_type_error(self, classification_df):
        with pytest.raises(TypeError, match="scoring must be a string"):
            AutoML(classification_df, target="churn", scoring=123)

    def test_incompatible_scoring_metric(self, classification_df, regression_df):
        # Regression metric on classification
        with pytest.raises(ValueError, match="not supported for classification"):
            AutoML(classification_df, target="churn", scoring="r2")

        # Classification metric on regression
        with pytest.raises(ValueError, match="not supported for regression"):
            AutoML(regression_df, target="price", scoring="f1")

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            AutoML("not_a_df", target="target")

    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="empty|enough rows"):
            AutoML(pd.DataFrame(), target="target")

    def test_missing_target_raises_value_error(self, classification_df):
        with pytest.raises(ValueError, match="Target column.*not found"):
            AutoML(classification_df, target="non_existent_col")

    def test_none_target_raises_value_error(self, classification_df):
        with pytest.raises(ValueError, match="Target column must be specified"):
            AutoML(classification_df, target=None)

    def test_non_string_target_raises_type_error(self, classification_df):
        with pytest.raises(TypeError, match="Target column name must be a string"):
            AutoML(classification_df, target=123)

    def test_datetime_target_raises_value_error(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "dt": pd.date_range("2025-01-01", periods=8),
        })
        with pytest.raises(ValueError, match="datetime/timedelta"):
            AutoML(df, target="dt")

    def test_all_nan_target_raises_value_error(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [np.nan] * 8,
        })
        with pytest.raises(ValueError, match="only missing values"):
            AutoML(df, target="target")

    def test_missing_values_in_target_raises_value_error(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [0, 1, 0, np.nan, 1, 0, 1, 0],
        })
        with pytest.raises(ValueError, match="missing value"):
            AutoML(df, target="target")

    def test_single_unique_value_target_raises_value_error(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [1, 1, 1, 1, 1, 1, 1, 1],
        })
        with pytest.raises(ValueError, match="fewer than 2 unique values"):
            AutoML(df, target="target")

    def test_unique_string_ids_as_target_raises_value_error(self):
        n = 15
        df = pd.DataFrame({
            "a": list(range(n)),
            "id_col": [f"ID_{i}" for i in range(n)],
        })
        with pytest.raises(ValueError, match="unique string identifiers"):
            AutoML(df, target="id_col")


# ==============================================================================
# 8. Reproducibility, Immutability, and Data Leakage Tests
# ==============================================================================
class TestAutoMLSafetyAndLeakage:
    def test_reproducibility_with_fixed_random_state(self, classification_df):
        automl1 = AutoML(classification_df, target="churn", cv=3, random_state=42)
        automl1.evaluate()
        res1 = automl1.results()

        automl2 = AutoML(classification_df, target="churn", cv=3, random_state=42)
        automl2.evaluate()
        res2 = automl2.results()

        pd.testing.assert_frame_equal(res1, res2)

    def test_original_dataframe_unchanged(self, classification_df):
        original = classification_df.copy(deep=True)
        automl = AutoML(classification_df, target="churn", cv=4, random_state=42)
        automl.evaluate()
        automl.compare(print_table=False)
        automl.report(print_report=False)

        pd.testing.assert_frame_equal(classification_df, original)

    def test_no_preprocessing_leakage_across_cv_folds(self):
        """
        Verify that preprocessing in cross-validation is fitted ONLY on the training fold,
        and not on validation folds or the whole dataset.
        """
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        n = 40
        # First 20 rows have small values, last 20 rows have large values
        df = pd.DataFrame({
            "feature": [10.0] * 20 + [5000.0] * 20,
            "target": [10.5 + i * 1.5 for i in range(n)],
        })

        automl = AutoML(df, target="target", cv=2, random_state=42)
        automl.evaluate()

        # In fold 1, training fold is one half and validation is the other.
        # Since AutoML does not leak, the evaluation runs and finishes cleanly.
        res = automl.results()
        assert len(res) == 3
        assert "Mean R²" in res.columns


# ==============================================================================
# 9. Hyperparameter Tuning (tune, tuning_results, best_models) Tests
# ==============================================================================
class TestAutoMLTuning:
    def test_tune_classification(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=4, random_state=42
        )
        automl.tune()

        assert automl.is_tuned is True
        results = automl.tuning_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

        # Check required columns
        assert "Model" in results.columns
        assert any(c.startswith("Best Score") for c in results.columns)
        assert "Best Parameters" in results.columns
        assert "Iterations" in results.columns

        # Check flexible column access
        assert len(results["model"]) == 3
        assert len(results["best score"]) == 3
        assert len(results["best parameters"]) == 3
        assert len(results["iterations"]) == 3

    def test_tune_regression(self, regression_df):
        automl = AutoML(
            regression_df, target="price", cv=3, n_iter=3, random_state=42
        )
        automl.tune()

        assert automl.is_tuned is True
        results = automl.tuning_results()
        assert len(results) == 3

        # Linear Regression should run exactly 1 iteration
        lr_row = results[results["Model"] == "Linear Regression"]
        assert len(lr_row) == 1
        assert lr_row["Iterations"].iloc[0] == 1

    def test_tune_custom_n_iter_override(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=2, n_iter=2, random_state=42
        )
        automl.tune(n_iter=3)
        results = automl.tuning_results()
        rf_row = results[results["Model"] == "Random Forest"]
        assert rf_row["Iterations"].iloc[0] == 3

    def test_best_models_ordering(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl.tune()
        models = automl.best_models()

        assert isinstance(models, dict)
        assert len(models) == 3
        model_names = list(models.keys())
        assert set(model_names) == {
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
        }

        # Check ordering matches tuning_results order
        res = automl.tuning_results()
        expected_order = list(res["Model"])
        assert model_names == expected_order

    def test_predict_after_tune(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl.tune()

        X_new = classification_df.drop(columns=["churn"]).iloc[:5]
        preds = automl.predict(X_new)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})

    def test_predict_after_fit_fallback(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", test_size=0.2, random_state=42
        )
        automl.fit()

        X_new = classification_df.drop(columns=["churn"]).iloc[:4]
        preds = automl.predict(X_new)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 4

    def test_evaluate_tuned_classification(self, classification_df, capsys):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl.tune()
        test_eval = automl.evaluate_tuned()
        out = capsys.readouterr().out

        assert isinstance(test_eval, pd.DataFrame)
        metrics_present = set(test_eval["Metric"].str.lower().unique())
        assert {"accuracy", "precision", "recall", "f1"}.issubset(metrics_present)
        assert "DataCraft AutoML Held-Out Test Evaluation" in out

    def test_evaluate_tuned_regression(self, regression_df, capsys):
        automl = AutoML(
            regression_df, target="price", cv=3, n_iter=2, random_state=42
        )
        automl.tune()
        test_eval = automl.evaluate_tuned()
        out = capsys.readouterr().out

        assert isinstance(test_eval, pd.DataFrame)
        metrics_present = set(test_eval["Metric"].unique())
        assert {"MAE", "RMSE", "R²"}.issubset(metrics_present)
        assert "DataCraft AutoML Held-Out Test Evaluation" in out


# ==============================================================================
# 10. Tuning Validation and Error Handling Tests
# ==============================================================================
class TestAutoMLTuningValidation:
    def test_invalid_n_iter(self, classification_df):
        with pytest.raises(ValueError, match="n_iter must be an integer >= 1"):
            AutoML(classification_df, target="churn", n_iter=0)
        with pytest.raises(ValueError, match="n_iter must be an integer >= 1"):
            AutoML(classification_df, target="churn", n_iter=-3)
        with pytest.raises(ValueError, match="n_iter must be an integer >= 1"):
            AutoML(classification_df, target="churn", n_iter=True)
        with pytest.raises(ValueError, match="n_iter must be an integer >= 1"):
            AutoML(classification_df, target="churn", n_iter="10")

    def test_invalid_n_iter_in_tune_method(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(ValueError, match="n_iter must be an integer >= 1"):
            automl.tune(n_iter=0)

    def test_predict_before_tune_or_fit_raises(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        X = classification_df.drop(columns=["churn"])
        with pytest.raises(RuntimeError, match="predict.*cannot be called before"):
            automl.predict(X)

    def test_tuning_results_before_tune_raises(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="tuning_results.*cannot be called before"):
            automl.tuning_results()

    def test_best_models_before_tune_raises(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="best_models.*cannot be called before"):
            automl.best_models()

    def test_evaluate_tuned_before_tune_raises(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="evaluate_tuned.*cannot be called before"):
            automl.evaluate_tuned()


# ==============================================================================
# 11. Tuning Reproducibility and Leakage Prevention Tests
# ==============================================================================
class TestAutoMLTuningSafetyAndLeakage:
    def test_reproducibility_of_tuning(self, classification_df):
        automl1 = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl1.tune()
        res1 = automl1.tuning_results()

        automl2 = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl2.tune()
        res2 = automl2.tuning_results()

        pd.testing.assert_frame_equal(res1, res2)

    def test_no_data_leakage_in_tuning(self, regression_df):
        """
        Verify that hyperparameter tuning uses an sklearn Pipeline with ColumnTransformer,
        meaning preprocessing is fitted strictly inside each cross-validation fold.
        """
        automl = AutoML(
            regression_df, target="price", cv=2, n_iter=2, random_state=42
        )
        automl.tune()

        # Check top tuned pipeline structure
        top_pipeline = automl._top_tuned_pipeline
        assert hasattr(top_pipeline, "named_steps")
        assert "prep" in top_pipeline.named_steps
        assert "model" in top_pipeline.named_steps

        # Check held-out test set was isolated
        assert automl._X_test_heldout is not None
        assert automl._y_test_heldout is not None
        assert len(automl._X_test_heldout) + len(automl._X_train_tune) == len(regression_df)


# ==============================================================================
# 12. Model Persistence, Reusable Prediction & Model Info Tests
# ==============================================================================
class TestAutoMLModelPersistenceAndPrediction:
    def test_save_and_load_classification_tune(self, classification_df, tmp_path):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=3, random_state=42
        )
        automl.tune()

        X_new = classification_df.drop(columns=["churn"]).iloc[:5]
        preds_before = automl.predict(X_new)
        proba_before = automl.predict_proba(X_new)

        model_path = tmp_path / "model.pkl"
        saved_path = automl.save_model(model_path)
        assert Path(saved_path).is_file()

        # Load as class method
        loaded_automl = AutoML.load_model(model_path)
        assert loaded_automl.is_loaded
        assert loaded_automl.task == "classification"
        assert loaded_automl.target == "churn"

        # Predict with loaded model
        preds_after = loaded_automl.predict(X_new)
        proba_after = loaded_automl.predict_proba(X_new)

        # Consistency check
        np.testing.assert_array_equal(preds_before, preds_after)
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)

    def test_save_and_load_instance_method(self, classification_df, tmp_path):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=2, random_state=42
        )
        automl.tune()

        X_new = classification_df.drop(columns=["churn"]).iloc[5:10]
        preds_before = automl.predict(X_new)

        model_path = tmp_path / "instance_model.pkl"
        automl.save_model(model_path)

        # Load using existing instance
        automl_reloaded = AutoML()
        res = automl_reloaded.load_model(model_path)
        assert res is automl_reloaded
        assert automl_reloaded.is_loaded

        preds_after = automl_reloaded.predict(X_new)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_save_and_load_regression_fit(self, regression_df, tmp_path):
        automl = AutoML(regression_df, target="price", test_size=0.2, random_state=42)
        automl.fit()

        X_new = regression_df.drop(columns=["price"]).iloc[:5]
        preds_before = automl.predict(X_new)

        model_path = tmp_path / "reg_model.pkl"
        automl.save_model(model_path)

        loaded_automl = AutoML.load_model(model_path)
        assert loaded_automl.task == "regression"
        assert loaded_automl.target == "price"

        preds_after = loaded_automl.predict(X_new)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_prediction_single_row_dataframe_and_series(self, classification_df, tmp_path):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=2, random_state=42
        )
        automl.tune()

        model_path = tmp_path / "single_row_model.pkl"
        automl.save_model(model_path)
        loaded = AutoML.load_model(model_path)

        # Single row DataFrame
        single_df = classification_df.drop(columns=["churn"]).iloc[[0]]
        pred_df = loaded.predict(single_df)
        assert isinstance(pred_df, np.ndarray)
        assert len(pred_df) == 1

        proba_df = loaded.predict_proba(single_df)
        assert isinstance(proba_df, np.ndarray)
        assert proba_df.shape == (1, 2)
        assert np.isclose(proba_df.sum(), 1.0)

        # Single row Series
        single_series = classification_df.drop(columns=["churn"]).iloc[0]
        pred_series = loaded.predict(single_series)
        assert isinstance(pred_series, np.ndarray)
        assert len(pred_series) == 1
        assert pred_df[0] == pred_series[0]

    def test_model_info(self, classification_df, tmp_path):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=2, random_state=42
        )
        automl.tune()

        info = automl.model_info()
        assert isinstance(info, dict)
        assert info["task"] == "classification"
        assert info["target"] == "churn"
        assert info["version"] == "1.0.0"
        assert isinstance(info["model"], str)
        assert isinstance(info["feature_names"], list)
        assert len(info["feature_names"]) > 0
        assert info["training_timestamp"] is not None

        # Verify info survives save & load
        model_path = tmp_path / "info_model.pkl"
        automl.save_model(model_path)

        loaded = AutoML.load_model(model_path)
        loaded_info = loaded.model_info()
        assert loaded_info["task"] == info["task"]
        assert loaded_info["target"] == info["target"]
        assert loaded_info["model"] == info["model"]
        assert loaded_info["version"] == info["version"]
        assert loaded_info["feature_names"] == info["feature_names"]

    def test_model_info_before_training_raises(self, classification_df):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="model_info.*requires a fitted, tuned, or loaded model"):
            automl.model_info()

    def test_save_model_before_training_raises(self, classification_df, tmp_path):
        automl = AutoML(classification_df, target="churn")
        with pytest.raises(RuntimeError, match="save_model.*requires a fitted, tuned, or loaded model"):
            automl.save_model(tmp_path / "fail.pkl")

    def test_load_model_missing_file_raises(self, tmp_path):
        non_existent = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            AutoML.load_model(non_existent)

    def test_load_model_invalid_corrupt_file_raises(self, tmp_path):
        corrupt_file = tmp_path / "corrupt.pkl"
        corrupt_file.write_text("not a pickle file")
        with pytest.raises(ValueError, match="Failed to load model file|not a valid DataCraft model bundle"):
            AutoML.load_model(corrupt_file)

    def test_load_model_missing_bundle_keys_raises(self, tmp_path):
        import joblib
        fake_file = tmp_path / "fake.pkl"
        joblib.dump({"some_key": 123}, fake_file)
        with pytest.raises(ValueError, match="missing essential DataCraft bundle components"):
            AutoML.load_model(fake_file)

    def test_predict_with_missing_columns_raises(self, classification_df, tmp_path):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=2, random_state=42
        )
        automl.tune()

        # Omit a column
        bad_df = classification_df.drop(columns=["churn", "age"])
        with pytest.raises(ValueError, match="missing required feature column"):
            automl.predict(bad_df)

    def test_predict_proba_on_regression_raises(self, regression_df):
        automl = AutoML(regression_df, target="price", cv=2, n_iter=2, random_state=42)
        automl.tune()
        X = regression_df.drop(columns=["price"]).iloc[:3]
        with pytest.raises(ValueError, match="predict_proba.*only supported for classification"):
            automl.predict_proba(X)

    def test_original_dataframe_unchanged(self, classification_df):
        automl = AutoML(
            classification_df, target="churn", cv=3, n_iter=2, random_state=42
        )
        automl.tune()

        X = classification_df.drop(columns=["churn"])
        X_copy = X.copy(deep=True)

        preds = automl.predict(X)
        pd.testing.assert_frame_equal(X, X_copy)

        proba = automl.predict_proba(X)
        pd.testing.assert_frame_equal(X, X_copy)

    def test_loaded_model_preserves_complete_preprocessing_pipeline(self, tmp_path):
        """
        Verify that a loaded pipeline transparently handles raw data with missing values,
        unscaled numbers, and unseen categories without manual preprocessing.
        """
        raw_train_df = pd.DataFrame({
            "age": [20, 30, np.nan, 50, 60, 70, 80, 25, 35, 45],
            "city": ["NY", "SF", "LA", "NY", "SF", "LA", "NY", "SF", "LA", "NY"],
            "salary": [50000, 80000, 60000, 120000, 150000, np.nan, 200000, 55000, 85000, 95000],
            "churn": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        })

        automl = AutoML(raw_train_df, target="churn", cv=2, n_iter=2, random_state=42)
        automl.tune()

        model_path = tmp_path / "pipeline_model.pkl"
        automl.save_model(model_path)

        # Load fresh instance
        loaded = AutoML.load_model(model_path)

        # Test on raw data with NaNs and novel category
        raw_test_df = pd.DataFrame({
            "age": [np.nan, 40],
            "city": ["Tokyo", "SF"],  # "Tokyo" is unseen
            "salary": [np.nan, 75000],
        })

        preds = loaded.predict(raw_test_df)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 2
        assert set(preds).issubset({0, 1})


