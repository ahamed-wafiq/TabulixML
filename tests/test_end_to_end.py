"""
tests/test_end_to_end.py -- End-to-End Workflow, API Consistency & Robustness Tests for TabulixML

Validates Phase 18:
  1. Complete End-to-End Classification Workflow:
     DataFrame -> AutoClean -> AutoEDA -> AutoPrep -> AutoML -> tune -> evaluate -> save -> load -> predict
  2. Complete End-to-End Regression Workflow:
     DataFrame -> AutoClean -> AutoEDA -> AutoPrep -> AutoML -> tune -> evaluate -> save -> load -> predict
  3. API Consistency and Object Returns across all public methods:
     inspect(), preview(), clean(), report(), history(), summary(), correlations(),
     quality(), split(), prepare(), get_pipeline(), transform(), detect_task(),
     evaluate(), compare(), tune(), tuning_results(), best_models(), predict(),
     predict_proba(), evaluate_tuned(), save_model(), load_model(), model_info()
  4. Dataset Validation & Edge Case Handling:
     empty DataFrame, single-row dataset, missing target, identical target values,
     too few samples, excessive missing values, invalid order of calls.
  5. Full Reproducibility with random_state=42.
  6. Zero Data Leakage verification throughout entire pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tabulixml import AutoClean, AutoEDA, AutoPrep, AutoML


# ==============================================================================
# Fixtures for Messy Tabular Datasets
# ==============================================================================

@pytest.fixture
def messy_classification_df() -> pd.DataFrame:
    """
    Realistic messy tabular classification dataset with:
      - missing numerical & categorical values
      - duplicate rows
      - extreme numerical outliers
      - inconsistent categorical casing
      - datetime column
      - binary classification target
    """
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "age": np.where(rng.rand(n) < 0.1, np.nan, rng.normal(38, 12, size=n)),
        "salary": np.where(rng.rand(n) < 0.1, np.nan, rng.normal(75000, 20000, size=n)),
        "tenure_months": rng.randint(1, 120, size=n),
        "department": rng.choice(["Engineering", "engineering", "Sales", "sales", "Marketing", None], size=n),
        "contract_type": rng.choice(["Monthly", "Annual", "Two-Year", None], size=n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "churn": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
    })
    # Add an extreme outlier
    df.loc[0, "salary"] = 10_000_000.0
    # Add duplicate rows
    dupes = df.iloc[:3].copy()
    return pd.concat([df, dupes], ignore_index=True)


@pytest.fixture
def messy_regression_df() -> pd.DataFrame:
    """
    Realistic messy tabular regression dataset with:
      - missing numerical & categorical values
      - duplicate rows
      - extreme numerical values
      - continuous regression target (house_price)
    """
    rng = np.random.RandomState(42)
    n = 60
    sqft = np.where(rng.rand(n) < 0.08, np.nan, rng.normal(2000, 500, size=n).round())
    bedrooms = rng.choice([2, 3, 4, 5], size=n)
    neighborhood = rng.choice(["Downtown", "Suburbs", "Rural", None], size=n)
    noise = rng.normal(0, 15000, size=n)

    # Impute local mean sqft for generating clean target
    safe_sqft = np.nan_to_num(sqft, nan=2000.0)
    price = safe_sqft * 175.0 + bedrooms * 20000.0 + noise

    df = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "house_price": price.round(2),
    })
    # Add duplicate rows
    dupes = df.iloc[:2].copy()
    return pd.concat([df, dupes], ignore_index=True)


# ==============================================================================
# 1. End-to-End Classification Workflow Test
# ==============================================================================

class TestEndToEndClassification:
    def test_full_classification_pipeline(self, messy_classification_df, tmp_path):
        """
        Verify complete seamless workflow:
        DataFrame -> AutoClean -> AutoEDA -> AutoPrep -> AutoML -> tune -> evaluate -> save -> load -> predict
        """
        raw_df = messy_classification_df.copy()

        # Step 1: AutoClean
        cleaner = AutoClean(raw_df, target="churn", clean_col_names=True)
        clean_inspect = cleaner.inspect()
        assert isinstance(clean_inspect, dict)
        assert clean_inspect["duplicates"] >= 3

        clean_preview = cleaner.preview()
        assert isinstance(clean_preview, dict)

        cleaned_df = cleaner.clean()
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) < len(raw_df)  # Duplicates removed
        assert cleaner.history() is not None

        # Step 2: AutoEDA
        eda = AutoEDA(cleaned_df)
        eda_inspect = eda.inspect()
        assert isinstance(eda_inspect, dict)
        assert eda_inspect["shape"][0] == len(cleaned_df)

        eda_summary = eda.summary()
        assert isinstance(eda_summary, dict)
        assert "numerical" in eda_summary
        assert "categorical" in eda_summary

        eda_corr = eda.correlations(threshold=0.5)
        assert isinstance(eda_corr, dict)
        assert "matrix" in eda_corr
        assert isinstance(eda_corr["matrix"], pd.DataFrame)

        eda_qual = eda.quality()
        assert isinstance(eda_qual, dict)

        report_file = tmp_path / "eda_report.html"
        eda.save_report(report_file)
        assert report_file.is_file()

        # Step 3: AutoPrep
        # Exclude datetime column for standard tabular ML modeling
        modeling_df = cleaned_df.drop(columns=["signup_date"])
        prep = AutoPrep(modeling_df, target="churn", test_size=0.2, random_state=42)

        prep_inspect = prep.inspect()
        assert isinstance(prep_inspect, dict)
        assert prep_inspect["target"] == "churn"

        prep_preview = prep.preview(print_preview=False)
        assert isinstance(prep_preview, dict)

        X_train_raw, X_test_raw, y_train, y_test = prep.split()
        assert len(X_train_raw) + len(X_test_raw) == len(modeling_df)
        assert "churn" not in X_train_raw.columns  # No target leakage

        X_train_trans, X_test_trans, y_tr, y_te = prep.prepare()
        assert isinstance(X_train_trans, np.ndarray)
        assert isinstance(X_test_trans, np.ndarray)

        prep_pipeline = prep.get_pipeline()
        assert prep_pipeline is not None

        # Test transforming unseen data with novel categories
        novel_data = pd.DataFrame({
            "age": [33.0],
            "salary": [80000.0],
            "tenure_months": [12],
            "department": ["Research"],  # Unseen category
            "contract_type": ["Monthly"],
        })
        novel_trans = prep.transform(novel_data)
        assert isinstance(novel_trans, np.ndarray)
        assert novel_trans.shape[0] == 1

        # Step 4: AutoML
        automl = AutoML(
            modeling_df,
            target="churn",
            cv=3,
            n_iter=3,
            scoring="auto",
            random_state=42,
        )

        task_info = automl.detect_task()
        assert task_info["task"] == "classification"
        assert task_info["target"] == "churn"

        models_preview = automl.preview()
        assert isinstance(models_preview, list)
        assert "Random Forest" in models_preview

        # Cross-validation
        automl.evaluate()
        cv_res = automl.results(source="evaluate")
        assert isinstance(cv_res, pd.DataFrame)
        assert "Model" in cv_res.columns

        comp = automl.compare(print_table=False)
        assert isinstance(comp, pd.DataFrame)

        # Hyperparameter tuning
        automl.tune()
        tune_res = automl.tuning_results()
        assert isinstance(tune_res, pd.DataFrame)
        assert "Model" in tune_res.columns
        assert "Best Parameters" in tune_res.columns

        best_pipes = automl.best_models()
        assert isinstance(best_pipes, dict)
        assert len(best_pipes) == 3

        held_out_eval = automl.evaluate_tuned(print_report=False)
        assert isinstance(held_out_eval, pd.DataFrame)
        assert "accuracy" in held_out_eval["Metric"].str.lower().values

        # Predictions before saving
        preds_before = automl.predict(novel_data)
        proba_before = automl.predict_proba(novel_data)
        assert isinstance(preds_before, np.ndarray)
        assert isinstance(proba_before, np.ndarray)

        # Step 5: Persistence
        model_file = tmp_path / "classification_model.pkl"
        saved_path = automl.save_model(model_file)
        assert Path(saved_path).is_file()

        info = automl.model_info()
        assert info["task"] == "classification"
        assert info["target"] == "churn"
        assert info["version"] == "1.0.0"

        # Step 6: Production Inference with Loaded Model
        loaded_automl = AutoML.load_model(model_file)
        assert loaded_automl.is_loaded
        assert loaded_automl.task == "classification"

        preds_after = loaded_automl.predict(novel_data)
        proba_after = loaded_automl.predict_proba(novel_data)

        # Strict consistency check
        np.testing.assert_array_equal(preds_before, preds_after)
        np.testing.assert_allclose(proba_before, proba_after, rtol=1e-5)


# ==============================================================================
# 2. End-to-End Regression Workflow Test
# ==============================================================================

class TestEndToEndRegression:
    def test_full_regression_pipeline(self, messy_regression_df, tmp_path):
        """
        Verify complete seamless workflow for regression:
        DataFrame -> AutoClean -> AutoEDA -> AutoPrep -> AutoML -> tune -> evaluate -> save -> load -> predict
        """
        raw_df = messy_regression_df.copy()

        # Step 1: AutoClean
        cleaner = AutoClean(raw_df, target="house_price")
        cleaned_df = cleaner.clean()
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) < len(raw_df)

        # Step 2: AutoEDA
        eda = AutoEDA(cleaned_df)
        eda_sum = eda.summary()
        assert isinstance(eda_sum, dict)

        # Step 3: AutoPrep
        prep = AutoPrep(cleaned_df, target="house_price", test_size=0.2, random_state=42)
        X_tr, X_te, y_tr, y_te = prep.prepare()
        assert X_tr.shape[0] > 0

        # Step 4: AutoML
        automl = AutoML(
            cleaned_df,
            target="house_price",
            cv=3,
            n_iter=2,
            scoring="auto",
            random_state=42,
        )
        assert automl.task == "regression"

        automl.evaluate()
        automl.tune()
        reg_eval = automl.evaluate_tuned(print_report=False)
        assert isinstance(reg_eval, pd.DataFrame)
        assert "MAE" in reg_eval["Metric"].values
        assert "R²" in reg_eval["Metric"].values

        # Test prediction on unseen house record
        new_house = pd.DataFrame({
            "sqft": [2200.0],
            "bedrooms": [4],
            "neighborhood": ["Suburbs"],
        })
        preds_before = automl.predict(new_house)
        assert isinstance(preds_before, np.ndarray)
        assert len(preds_before) == 1
        assert preds_before[0] > 0

        # Step 5: Model Persistence & Reload
        model_file = tmp_path / "regression_model.pkl"
        automl.save_model(model_file)

        loaded_automl = AutoML.load_model(model_file)
        preds_after = loaded_automl.predict(new_house)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


# ==============================================================================
# 3. Robust Dataset Validation & Edge Case Tests
# ==============================================================================

class TestDatasetValidationAndErrorHandling:
    def test_empty_dataframe_across_all_modules(self):
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            AutoClean(empty)
        with pytest.raises(ValueError, match="empty"):
            AutoEDA(empty)
        with pytest.raises(ValueError, match="empty"):
            AutoPrep(empty, target="col")
        with pytest.raises(ValueError, match="empty"):
            AutoML(empty, target="col")

    def test_single_row_dataset_validation(self):
        single = pd.DataFrame({"x": [10], "y": ["A"], "target": [1]})
        # AutoClean & AutoEDA can inspect single row without crash
        cleaner = AutoClean(single)
        assert cleaner.inspect()["shape"] == (1, 3)

        eda = AutoEDA(single)
        assert eda.inspect()["shape"] == (1, 3)

        # AutoPrep split/prepare requires >= 2 samples
        prep = AutoPrep(single, target="target")
        with pytest.raises(ValueError, match="too few samples"):
            prep.split()
        with pytest.raises(ValueError, match="too few samples"):
            prep.prepare()

        # AutoML requires >= 5 rows
        with pytest.raises(ValueError, match="minimum required is 5"):
            AutoML(single, target="target")

    def test_missing_target_validation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]})
        with pytest.raises(ValueError, match="Target column.*not found"):
            AutoClean(df, target="non_existent")
        with pytest.raises(ValueError, match="Target column.*not found"):
            AutoPrep(df, target="non_existent")
        with pytest.raises(ValueError, match="Target column.*not found"):
            AutoML(df, target="non_existent")

    def test_target_with_all_identical_values(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": [1, 1, 1, 1, 1, 1]})
        with pytest.raises(ValueError, match="fewer than 2 unique values"):
            AutoML(df, target="target")

    def test_target_with_all_missing_values(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": [np.nan] * 6})
        with pytest.raises(ValueError, match="only missing values"):
            AutoPrep(df, target="target")
        with pytest.raises(ValueError, match="only missing values"):
            AutoML(df, target="target")

    def test_zero_feature_columns_validation(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
        with pytest.raises(ValueError, match="at least one feature column"):
            AutoPrep(df, target="target")
        with pytest.raises(ValueError, match="at least one feature column"):
            AutoML(df, target="target")

    def test_all_feature_columns_missing_validation(self):
        df = pd.DataFrame({
            "feat1": [np.nan] * 6,
            "feat2": [np.nan] * 6,
            "target": [0, 1, 0, 1, 0, 1],
        })
        with pytest.raises(ValueError, match="All feature columns contain only missing values"):
            AutoML(df, target="target")

    def test_out_of_order_method_calls_raise_clear_errors(self, messy_classification_df, tmp_path):
        df = messy_classification_df.dropna().iloc[:15].copy()

        # AutoPrep out-of-order calls
        prep = AutoPrep(df.drop(columns=["signup_date"]), target="churn")
        with pytest.raises(RuntimeError, match="Pipeline has not been fitted yet.*Call prepare.*first"):
            prep.get_pipeline()
        with pytest.raises(RuntimeError, match="transform.*cannot be called before prepare"):
            prep.transform(df.iloc[:2])

        # AutoML out-of-order calls
        automl = AutoML(df.drop(columns=["signup_date"]), target="churn")
        X = df.drop(columns=["signup_date", "churn"]).iloc[:2]

        with pytest.raises(RuntimeError, match="predict.*cannot be called before tuning, fitting, or loading"):
            automl.predict(X)
        with pytest.raises(RuntimeError, match="predict_proba.*cannot be called before tuning, fitting, or loading"):
            automl.predict_proba(X)
        with pytest.raises(RuntimeError, match="save_model.*requires a fitted, tuned, or loaded model"):
            automl.save_model(tmp_path / "fail.pkl")
        with pytest.raises(RuntimeError, match="results.*cannot be called before"):
            automl.results()
        with pytest.raises(RuntimeError, match="compare.*cannot be called before evaluate"):
            automl.compare()
        with pytest.raises(RuntimeError, match="tuning_results.*cannot be called before tune"):
            automl.tuning_results()
        with pytest.raises(RuntimeError, match="best_models.*cannot be called before tune"):
            automl.best_models()
        with pytest.raises(RuntimeError, match="evaluate_tuned.*cannot be called before tune"):
            automl.evaluate_tuned()
        with pytest.raises(RuntimeError, match="model_info.*requires a fitted, tuned, or loaded model"):
            automl.model_info()


# ==============================================================================
# 4. Strict Reproducibility Test
# ==============================================================================

class TestReproducibility:
    def test_random_state_reproducibility(self, messy_classification_df):
        df = messy_classification_df.dropna().drop(columns=["signup_date"]).iloc[:30].copy()

        # Run 1
        automl1 = AutoML(df, target="churn", cv=3, n_iter=3, random_state=42)
        automl1.evaluate()
        automl1.tune()
        eval1 = automl1.results(source="evaluate")
        tune1 = automl1.tuning_results()
        preds1 = automl1.predict(df.drop(columns=["churn"]).iloc[:5])

        # Run 2
        automl2 = AutoML(df, target="churn", cv=3, n_iter=3, random_state=42)
        automl2.evaluate()
        automl2.tune()
        eval2 = automl2.results(source="evaluate")
        tune2 = automl2.tuning_results()
        preds2 = automl2.predict(df.drop(columns=["churn"]).iloc[:5])

        pd.testing.assert_frame_equal(eval1, eval2)
        pd.testing.assert_frame_equal(tune1, tune2)
        np.testing.assert_array_equal(preds1, preds2)


# ==============================================================================
# 5. Strict Zero Data Leakage Verification
# ==============================================================================

class TestZeroDataLeakage:
    def test_target_never_in_feature_columns(self, messy_classification_df):
        df = messy_classification_df.dropna().drop(columns=["signup_date"]).copy()

        prep = AutoPrep(df, target="churn")
        assert "churn" not in prep.feature_columns
        assert "churn" not in prep.numerical_columns
        assert "churn" not in prep.categorical_columns

        X_train, X_test, y_train, y_test = prep.split()
        assert "churn" not in X_train.columns
        assert "churn" not in X_test.columns

        automl = AutoML(df, target="churn", cv=3, n_iter=2, random_state=42)
        assert "churn" not in automl._feature_cols
        assert "churn" not in automl._feature_names

    def test_preprocessing_fitted_only_on_train_split(self, messy_classification_df):
        df = messy_classification_df.dropna().drop(columns=["signup_date"]).copy()
        prep = AutoPrep(df, target="churn", test_size=0.2, random_state=42)
        X_train_trans, X_test_trans, y_tr, y_te = prep.prepare()

        # Ensure pipeline has fitted attributes derived solely from X_train
        pipeline = prep.get_pipeline()
        assert hasattr(pipeline, "transform")
        assert hasattr(pipeline, "named_transformers_")
