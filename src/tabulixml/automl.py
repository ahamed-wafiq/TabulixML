"""
src/tabulixml/automl.py -- AutoML Module for TabulixML

Provides lightweight, baseline automated machine learning with leak-free cross-validation
and controlled hyperparameter tuning:
  - Task detection: automatically detects 'classification' or 'regression' from the target column.
  - inspect(): displays target distribution, task type, sample counts, feature types, and missing values.
  - preview(): shows the baseline models that will be tested without running training.
  - fit(): uses AutoPrep internally to split data and fit preprocessing strictly on X_train (preventing
           target leakage), then trains baseline models on the single train/test split.
  - evaluate(): performs K-Fold (regression) or StratifiedKFold (classification) cross-validation.
                Ensures zero preprocessing leakage by fitting ColumnTransformer strictly on the training fold
                of each iteration.
  - compare(): displays a clean comparison table sorted by the primary evaluation metric, labeling the
               highest-scoring model.
  - tune(): performs controlled hyperparameter search using sklearn RandomizedSearchCV inside leak-free
            Pipelines.
  - tuning_results(): returns a DataFrame with model, best score, best parameters, and iterations.
  - best_models(): returns the tuned models ordered by highest cross-validation score for the selected metric.
  - predict(X): generates predictions using the tuned pipeline (or fitted model).
  - evaluate_tuned(): evaluates the top tuned model on the held-out test set (accuracy, precision, recall, F1
                      or MAE, RMSE, R²).
  - results(): returns a pandas DataFrame with evaluation scores.
  - report(): prints and returns a comprehensive, readable summary of AutoML execution.

Baseline models:
  - Classification: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
  - Regression: LinearRegression, DecisionTreeRegressor, RandomForestRegressor

Guarantees:
  - Strict immutability: never modifies the caller's DataFrame in place.
  - Zero target & test leakage: preprocessing pipelines are fitted strictly per training partition/fold.
  - Reproducibility: propagates random_state to all stochastic models, splits, and search iterations.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabulixml.cleaner import AutoClean
from tabulixml.prep import AutoPrep

TABULIXML_VERSION = "1.0.0"


class _ClassOrInstanceMethod:
    """Descriptor that allows a method to be invoked on either the class or an instance."""

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, instance, owner):
        if instance is None:
            return lambda *args, **kwargs: self.fn(owner, *args, **kwargs)
        return lambda *args, **kwargs: self.fn(instance, *args, **kwargs)


class FlexibleColumnIndex(pd.Index):
    """
    A pandas Index that allows case-insensitive column lookups for convenience.
    """

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            lower_self = [str(c).lower() for c in self]
            if key.lower() in lower_self:
                return True
            # Also support standard metric names or aliases
            aliases = {
                "model name": "model",
                "evaluation metric": "metric",
                "evaluation_metric": "metric",
                "standard deviation": "std",
                "standard_deviation": "std",
                "best score": "best score",
                "best parameter": "best parameters",
                "best parameters": "best parameters",
                "parameters": "best parameters",
                "params": "best parameters",
                "iterations": "iterations",
                "number of iterations": "iterations",
                "n_iter": "iterations",
            }
            if aliases.get(key.lower()) in lower_self:
                return True
            if key.lower() in ["mean", "mean score", "mean_score"]:
                if any(str(c).lower().startswith("mean") for c in self):
                    return True
            if key.lower() in ["best", "best score", "score"]:
                if any(str(c).lower().startswith("best") for c in self):
                    return True
            # Known metric lookups
            if key.lower() in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mae",
                "rmse",
                "r2",
                "r²",
            ]:
                return True
            # Fold lookups like "fold 1" or "fold_1"
            fold_key = key.lower().replace("_", " ")
            if fold_key in lower_self:
                return True
        return False


class ResultsDataFrame(pd.DataFrame):
    """
    A pandas DataFrame subclass returned by AutoML methods that supports
    flexible case-insensitive column access and metric lookups.
    """

    @property
    def _constructor(self):
        return ResultsDataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "columns") and len(self.columns) > 0:
            self.columns = FlexibleColumnIndex(self.columns)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in list(self.columns):
                return super().__getitem__(key)
            lower_map = {str(c).lower(): c for c in self.columns}
            if key.lower() in lower_map:
                return super().__getitem__(lower_map[key.lower()])
            aliases = {
                "model name": "model",
                "evaluation metric": "metric",
                "evaluation_metric": "metric",
                "standard deviation": "std",
                "standard_deviation": "std",
                "parameters": "best parameters",
                "params": "best parameters",
                "iterations": "iterations",
                "n_iter": "iterations",
            }
            if key.lower() in aliases and aliases[key.lower()] in lower_map:
                return super().__getitem__(lower_map[aliases[key.lower()]])

            # Check if key is looking for "mean" or "mean score" in CV table
            if key.lower() in ["mean", "mean score", "mean_score"]:
                for col in self.columns:
                    if str(col).lower().startswith("mean"):
                        return super().__getitem__(col)

            # Check if key is looking for "best score" in tuning table
            if key.lower() in ["best score", "best_score", "score"]:
                for col in self.columns:
                    if str(col).lower().startswith("best score"):
                        return super().__getitem__(col)

            # In long-format fit table: if user queries a metric like df["accuracy"] or df["MAE"]
            norm_key = key.lower().replace("r2", "r²")
            if "metric" in lower_map and "score" in lower_map:
                metric_col = lower_map["metric"]
                score_col = lower_map["score"]
                mask = self[metric_col].astype(str).str.lower() == norm_key
                if mask.any():
                    return self.loc[mask, score_col]
        return super().__getitem__(key)


class AutoML:
    """
    Automated Machine Learning exploration, cross-validation, and hyperparameter tuning for tabular data.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing features and target. Must be non-empty.
    target : str
        Target column name to predict. Must exist in df and contain no NaNs.
    test_size : float, default 0.2
        Fraction of dataset to allocate to test partition for fit() and tune() (0.0 < test_size < 1.0).
    cv : int, default 5
        Number of cross-validation folds for evaluate() and tune() (cv >= 2).
    n_iter : int, default 10
        Number of parameter settings sampled in RandomizedSearchCV during tune() (n_iter >= 1).
    scoring : str, default 'auto'
        Scoring metric for cross-validation evaluation and tuning.
        If 'auto': 'f1' for classification, 'r2' for regression.
    random_state : int or None, default 42
        Random seed for reproducible train/test splits, CV folds, and model initialization.

    Examples
    --------
    >>> import pandas as pd
    >>> from tabulixml import AutoML
    >>> df = pd.DataFrame({
    ...     "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    ...     "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    ...     "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> automl = AutoML(df, target="churn", cv=3, n_iter=5, random_state=42)
    >>> automl.evaluate()
    >>> automl.compare()
    >>> automl.tune()
    >>> print(automl.tuning_results())
    >>> print(automl.evaluate_tuned())
    >>> predictions = automl.predict(df.drop(columns=["churn"]))
    """

    # Supported scoring metric aliases
    _CLASSIFICATION_METRICS = {
        "auto",
        "f1",
        "f1_weighted",
        "accuracy",
        "precision",
        "precision_weighted",
        "recall",
        "recall_weighted",
        "roc_auc",
    }
    _REGRESSION_METRICS = {
        "auto",
        "r2",
        "r²",
        "mae",
        "rmse",
        "mse",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_error",
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        test_size: float = 0.2,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str = "auto",
        random_state: Optional[int] = 42,
    ) -> None:
        if df is None and target is None:
            self._init_empty()
            self._test_size = float(test_size)
            self._cv = int(cv)
            self._n_iter = int(n_iter)
            self._user_scoring = scoring
            self._scoring_param = scoring
            self._random_state = random_state
            return

        if df is None and target is not None:
            raise ValueError("Input DataFrame must be provided.")

        if target is None:
            raise ValueError("Target column must be specified.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"AutoML expects a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if len(df) < 5:
            raise ValueError(
                f"Dataset must contain enough rows for AutoML train/test splitting "
                f"(found {len(df)} rows, minimum required is 5)."
            )

        if not isinstance(target, str):
            raise TypeError(f"Target column name must be a string, got {type(target).__name__}.")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        target_series = df[target]

        # Check for datetime / timedelta target
        if (
            pd.api.types.is_datetime64_any_dtype(target_series)
            or pd.api.types.is_timedelta64_dtype(target_series)
        ):
            raise ValueError(
                f"Target column '{target}' is a datetime/timedelta type, "
                f"which is not supported as a machine learning target."
            )

        # Check for all-missing target
        if target_series.isna().all():
            raise ValueError(f"Target column '{target}' contains only missing values.")

        # Check for missing values in target
        if target_series.isna().any():
            missing_count = int(target_series.isna().sum())
            raise ValueError(
                f"Target column '{target}' contains {missing_count} missing value(s). "
                f"AutoML requires targets without missing values. "
                f"Please clean target missing values with AutoClean before running AutoML."
            )

        non_null_target = target_series.dropna()
        n_unique = non_null_target.nunique()

        # Target must have at least 2 distinct values
        if n_unique < 2:
            raise ValueError(
                f"Target column '{target}' has fewer than 2 unique values ({n_unique}). "
                f"Machine learning requires at least 2 distinct target values."
            )

        # Check unhashable types (list, dict, set)
        if non_null_target.apply(lambda x: isinstance(x, (list, dict, set))).any():
            raise ValueError(
                f"Target column '{target}' contains unhashable collection types (list/dict/set), "
                f"which are not supported as a machine learning target."
            )

        # Check for high-cardinality string IDs where every row is distinct
        if (
            (non_null_target.dtype == object or pd.api.types.is_string_dtype(non_null_target))
            and n_unique == len(df)
            and len(df) > 10
        ):
            raise ValueError(
                f"Target column '{target}' contains unique string identifiers for every row, "
                f"which cannot be used as a machine learning target."
            )

        # Validate test_size
        if not isinstance(test_size, (int, float)) or not (0.0 < float(test_size) < 1.0):
            raise ValueError(
                f"test_size must be a float between 0.0 and 1.0 (exclusive), got {test_size}."
            )

        # Validate cv
        if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {cv}.")

        # Validate n_iter
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError(f"n_iter must be an integer >= 1, got {n_iter}.")

        # Validate random_state
        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0:
                raise ValueError(
                    f"random_state must be a non-negative integer or None, got {random_state}."
                )

        self._df = df
        self._target = target
        self._test_size = float(test_size)
        self._cv = int(cv)
        self._n_iter = int(n_iter)
        self._random_state = random_state

        # Automatically detect task upfront for validation of scoring
        self._task: Optional[str] = None
        self.detect_task()

        # Validate scoring metric
        if not isinstance(scoring, str):
            raise TypeError(f"scoring must be a string, got {type(scoring).__name__}.")

        scoring_clean = scoring.strip().lower()
        if self._task == "classification":
            if scoring_clean not in self._CLASSIFICATION_METRICS:
                if scoring_clean in (self._REGRESSION_METRICS - {"auto"}):
                    raise ValueError(
                        f"Scoring metric '{scoring}' is not supported for classification tasks."
                    )
                raise ValueError(
                    f"Invalid scoring metric '{scoring}'. Supported classification metrics: "
                    f"{sorted(self._CLASSIFICATION_METRICS)}."
                )
            self._primary_metric = "f1" if scoring_clean == "auto" else scoring_clean
            self._primary_metric_label = (
                "F1"
                if self._primary_metric in ["f1", "f1_weighted"]
                else self._primary_metric.capitalize()
            )
        else:
            if scoring_clean not in self._REGRESSION_METRICS:
                if scoring_clean in (self._CLASSIFICATION_METRICS - {"auto"}):
                    raise ValueError(
                        f"Scoring metric '{scoring}' is not supported for regression tasks."
                    )
                raise ValueError(
                    f"Invalid scoring metric '{scoring}'. Supported regression metrics: "
                    f"{sorted(self._REGRESSION_METRICS)}."
                )
            self._primary_metric = "r2" if scoring_clean == "auto" else scoring_clean
            self._primary_metric_label = (
                "R²"
                if self._primary_metric in ["r2", "r²"]
                else (
                    "RMSE"
                    if "rmse" in self._primary_metric
                    else (
                        "MAE"
                        if "mae" in self._primary_metric
                        else self._primary_metric.upper()
                    )
                )
            )

        self._user_scoring = scoring

        # Identify feature types
        col_types = AutoClean._detect_col_types(self._df)
        self._feature_cols = [c for c in self._df.columns if c != self._target]
        if len(self._feature_cols) == 0:
            raise ValueError(
                "DataFrame must contain at least one feature column besides the target column."
            )
        if all(self._df[c].isna().all() for c in self._feature_cols):
            raise ValueError(
                "All feature columns contain only missing values. Please clean data with AutoClean before AutoML."
            )
        self._feature_names: list[str] = list(self._feature_cols)
        self._num_cols = [c for c in self._feature_cols if col_types.get(c) == "numerical"]
        self._cat_cols = [c for c in self._feature_cols if col_types.get(c) == "categorical"]

        # Single train/test fit tracking
        self._prep: Optional[AutoPrep] = None
        self._fitted_models: dict[str, Any] = {}
        self._scores: dict[str, dict[str, float]] = {}
        self._is_fitted: bool = False
        self._results_df: Optional[ResultsDataFrame] = None

        # Cross-validation tracking
        self._is_evaluated: bool = False
        self._cv_scores: dict[str, list[float]] = {}
        self._cv_results_df: Optional[ResultsDataFrame] = None

        # Tuning tracking
        self._is_tuned: bool = False
        self._is_loaded: bool = False
        self._is_empty: bool = False
        self._pipeline: Optional[Pipeline] = None
        self._training_timestamp: Optional[str] = None
        self._best_score: Optional[float] = None
        self._metadata: dict[str, Any] = {}
        self._tuned_searches: dict[str, dict[str, Any]] = {}
        self._top_tuned_pipeline: Optional[Pipeline] = None
        self._top_tuned_name: Optional[str] = None
        self._tuning_results_df: Optional[ResultsDataFrame] = None
        self._X_train_tune: Optional[pd.DataFrame] = None
        self._X_test_heldout: Optional[pd.DataFrame] = None
        self._y_train_tune: Optional[pd.Series] = None
        self._y_test_heldout: Optional[pd.Series] = None

    def _init_empty(self) -> None:
        """Initialize an unconfigured AutoML instance ready for model loading."""
        self._df: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None
        self._task: Optional[str] = None
        self._feature_cols: list[str] = []
        self._feature_names: list[str] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._test_size: float = 0.2
        self._cv: int = 5
        self._n_iter: int = 10
        self._user_scoring: str = "auto"
        self._scoring_param: str = "auto"
        self._primary_metric: str = "auto"
        self._primary_metric_label: str = "Score"
        self._random_state: Optional[int] = 42
        self._is_empty: bool = True
        self._is_fitted: bool = False
        self._is_evaluated: bool = False
        self._is_tuned: bool = False
        self._is_loaded: bool = False
        self._pipeline: Optional[Pipeline] = None
        self._top_tuned_pipeline: Optional[Pipeline] = None
        self._top_tuned_name: Optional[str] = None
        self._training_timestamp: Optional[str] = None
        self._best_score: Optional[float] = None
        self._fitted_models: dict[str, Any] = {}
        self._scores: dict[str, dict[str, float]] = {}
        self._results_df: Optional[ResultsDataFrame] = None
        self._cv_scores: dict[str, list[float]] = {}
        self._cv_results_df: Optional[ResultsDataFrame] = None
        self._tuned_searches: dict[str, dict[str, Any]] = {}
        self._tuning_results_df: Optional[ResultsDataFrame] = None
        self._X_train_tune: Optional[pd.DataFrame] = None
        self._X_test_heldout: Optional[pd.DataFrame] = None
        self._y_train_tune: Optional[pd.Series] = None
        self._y_test_heldout: Optional[pd.Series] = None
        self._prep: Optional[AutoPrep] = None
        self._metadata: dict[str, Any] = {}

    @property
    def target(self) -> Optional[str]:
        """Return the target column name."""
        return self._target

    @property
    def is_loaded(self) -> bool:
        """Return whether a model pipeline was loaded from file."""
        return self._is_loaded

    @property
    def task(self) -> str:
        """Return the detected task ('classification' or 'regression')."""
        if self._task is None:
            if self._df is not None and self._target is not None:
                self.detect_task()
            else:
                return "unknown"
        return self._task  # type: ignore

    @property
    def is_fitted(self) -> bool:
        """Return whether single train/test fit() has been completed."""
        return self._is_fitted

    @property
    def is_evaluated(self) -> bool:
        """Return whether cross-validation evaluate() has been completed."""
        return self._is_evaluated

    @property
    def is_tuned(self) -> bool:
        """Return whether hyperparameter tuning tune() has been completed."""
        return self._is_tuned

    @property
    def cv(self) -> int:
        """Return the number of cross-validation folds."""
        return self._cv

    @property
    def n_iter(self) -> int:
        """Return the number of search iterations for tuning."""
        return self._n_iter

    @property
    def scoring(self) -> str:
        """Return the configured scoring metric."""
        return self._user_scoring

    def detect_task(self) -> dict[str, str]:
        """
        Automatically determine whether the task is 'classification' or 'regression'
        based on the target column data type and value characteristics.

        Returns
        -------
        dict[str, str]
            {'task': 'classification' | 'regression', 'target': target_column_name}
        """
        if self._df is None or self._target is None:
            if self._task is not None:
                return {"task": self._task, "target": str(self._target or "unknown")}
            raise RuntimeError(
                "detect_task() requires an initialized DataFrame and target. "
                "Please instantiate AutoML with df and target."
            )
        target_series = self._df[self._target]

        # 1. Non-numeric types (boolean, string, object, categorical) -> classification
        if (
            pd.api.types.is_bool_dtype(target_series)
            or isinstance(target_series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(target_series)
            or target_series.dtype == object
        ):
            task = "classification"

        # 2. Floating-point numbers
        elif pd.api.types.is_float_dtype(target_series):
            unique_vals = target_series.dropna().unique()
            # If float has fractional parts or high cardinality -> regression
            has_fractional = np.any(np.mod(unique_vals, 1) != 0)
            if has_fractional or len(unique_vals) > 10:
                task = "regression"
            elif len(unique_vals) == 2:
                # e.g., binary encoded as 0.0 and 1.0
                task = "classification"
            else:
                task = "classification"

        # 3. Integer numbers
        elif pd.api.types.is_integer_dtype(target_series):
            unique_vals = target_series.dropna().unique()
            # Binary or small discrete set of classes (<= 10 classes)
            if len(unique_vals) <= 10:
                task = "classification"
            else:
                task = "regression"

        # 4. Fallback based on unique value count
        else:
            if target_series.nunique() <= 10:
                task = "classification"
            else:
                task = "regression"

        self._task = task
        return {"task": task, "target": self._target}

    def inspect(self) -> dict[str, Any]:
        """
        Inspect the dataset and target column characteristics.

        Shows:
          - target column
          - task type
          - number of samples
          - number of features
          - numerical features
          - categorical features
          - target distribution
          - missing values
          - class count for classification

        Returns
        -------
        dict[str, Any]
            Dictionary containing the inspected dataset and target properties.
        """
        task_info = self.detect_task()
        task = task_info["task"]
        target_series = self._df[self._target]

        missing_vals = int(self._df.isna().sum().sum())
        target_missing = int(target_series.isna().sum())

        if task == "classification":
            val_counts = target_series.value_counts(dropna=False).to_dict()
            class_counts = {str(k): int(v) for k, v in val_counts.items()}
            target_distribution = class_counts
        else:
            class_counts = None
            target_distribution = {
                "mean": round(float(target_series.mean()), 4),
                "std": round(float(target_series.std()), 4) if len(target_series) > 1 else 0.0,
                "min": round(float(target_series.min()), 4),
                "max": round(float(target_series.max()), 4),
            }

        info: dict[str, Any] = {
            "target": self._target,
            "task": task,
            "samples": len(self._df),
            "features": len(self._feature_cols),
            "numerical_features": list(self._num_cols),
            "categorical_features": list(self._cat_cols),
            "target_distribution": target_distribution,
            "missing_values": missing_vals,
            "class_counts": class_counts,
        }

        # Print human-readable summary
        print("=" * 60)
        print("TabulixML AutoML Inspection")
        print("=" * 60)
        print(f"Target Column        : {self._target}")
        print(f"Task Type            : {task}")
        print(f"Number of Samples    : {len(self._df)}")
        print(f"Number of Features   : {len(self._feature_cols)}")
        print(f"Numerical Features   : {self._num_cols}")
        print(f"Categorical Features : {self._cat_cols}")
        print(f"Missing Values       : {missing_vals} total (Target missing: {target_missing})")
        if task == "classification":
            print(f"Class Counts         : {class_counts}")
        else:
            print(f"Target Distribution  : {target_distribution}")
        print("=" * 60)

        return info

    def preview(self) -> list[str]:
        """
        Show which baseline models will be tested without executing any training.

        Returns
        -------
        list[str]
            List of model names to be tested.
        """
        task_info = self.detect_task()
        task = task_info["task"]

        if task == "classification":
            models = ["Logistic Regression", "Decision Tree", "Random Forest"]
        else:
            models = ["Linear Regression", "Decision Tree", "Random Forest"]

        print("TabulixML AutoML Preview")
        print(f"Task: {task}")
        print("Models:")
        for m in models:
            print(f"* {m}")

        return models

    # --------------------------------------------------------------------------
    # Helper: Preprocessing Pipeline Builder
    # --------------------------------------------------------------------------

    def _build_preprocessor(self) -> ColumnTransformer:
        """Construct a fresh, un-fitted ColumnTransformer pipeline."""
        transformers: list[tuple[str, Any, list[str]]] = []
        if self._num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, list(self._num_cols)))

        if self._cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, list(self._cat_cols)))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # --------------------------------------------------------------------------
    # Single Train/Test Split Training (fit)
    # --------------------------------------------------------------------------

    def fit(self) -> "AutoML":
        """
        Execute single train/test split workflow:
          1. Splits data into train and test sets using test_size and random_state.
          2. Fits preprocessing strictly on X_train (zero target leakage).
          3. Trains baseline models on (X_train, y_train).
          4. Evaluates models on (X_test, y_test).

        Returns
        -------
        AutoML
            Fitted AutoML instance.
        """
        task = self.detect_task()["task"]

        # Delegate leak-free preprocessing entirely to AutoPrep
        self._prep = AutoPrep(
            df=self._df,
            target=self._target,
            test_size=self._test_size,
            random_state=self._random_state,
        )
        X_train, X_test, y_train, y_test = self._prep.prepare()

        if task == "classification":
            models: dict[str, Any] = {
                "Logistic Regression": LogisticRegression(
                    random_state=self._random_state,
                    max_iter=1000,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=self._random_state,
                ),
                "Random Forest": RandomForestClassifier(
                    random_state=self._random_state,
                ),
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=self._random_state,
                ),
                "Random Forest": RandomForestRegressor(
                    random_state=self._random_state,
                ),
            }

        self._fitted_models = {}
        self._scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self._fitted_models[name] = model

            if task == "classification":
                acc = float(accuracy_score(y_test, y_pred))
                unique_train = np.unique(y_train)
                if len(unique_train) == 2 and set(unique_train).issubset({0, 1, True, False}):
                    prec = float(precision_score(y_test, y_pred, zero_division=0))
                    rec = float(recall_score(y_test, y_pred, zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, zero_division=0))
                else:
                    prec = float(
                        precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    )
                    rec = float(
                        recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    )
                    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

                self._scores[name] = {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "F1": round(f1, 4),
                }
            else:
                mae = float(mean_absolute_error(y_test, y_pred))
                mse = float(mean_squared_error(y_test, y_pred))
                rmse = float(np.sqrt(mse))
                r2 = float(r2_score(y_test, y_pred))

                self._scores[name] = {
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "R²": round(r2, 4),
                }

        # Build results DataFrame
        records = []
        for model_name, metrics in self._scores.items():
            for metric_name, score in metrics.items():
                records.append({
                    "model": model_name,
                    "metric": metric_name,
                    "score": score,
                })

        self._results_df = ResultsDataFrame(records)
        self._is_fitted = True

        top_name = list(self._fitted_models.keys())[0]
        top_model = self._fitted_models[top_name]
        self._top_tuned_name = top_name
        self._top_tuned_pipeline = Pipeline([
            ("prep", self._prep.get_pipeline()),
            ("model", top_model),
        ])
        self._pipeline = self._top_tuned_pipeline
        self._training_timestamp = datetime.now(timezone.utc).isoformat()
        first_metric = list(self._scores[top_name].keys())[0]
        self._best_score = float(self._scores[top_name][first_metric])
        return self

    # --------------------------------------------------------------------------
    # Leak-Free Cross-Validation Evaluation (evaluate)
    # --------------------------------------------------------------------------

    def _compute_metric_score(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: np.ndarray,
        metric_name: str,
    ) -> float:
        """Helper to calculate individual metric score."""
        m = metric_name.lower().replace("r²", "r2")
        if m == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif m in ["precision", "precision_weighted"]:
            unique_classes = np.unique(y_true)
            if (
                len(unique_classes) == 2
                and set(unique_classes).issubset({0, 1, True, False})
                and m == "precision"
            ):
                return float(precision_score(y_true, y_pred, zero_division=0))
            return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        elif m in ["recall", "recall_weighted"]:
            unique_classes = np.unique(y_true)
            if (
                len(unique_classes) == 2
                and set(unique_classes).issubset({0, 1, True, False})
                and m == "recall"
            ):
                return float(recall_score(y_true, y_pred, zero_division=0))
            return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        elif m in ["f1", "f1_weighted"]:
            unique_classes = np.unique(y_true)
            if (
                len(unique_classes) == 2
                and set(unique_classes).issubset({0, 1, True, False})
                and m == "f1"
            ):
                return float(f1_score(y_true, y_pred, zero_division=0))
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        elif m in ["mae", "neg_mean_absolute_error"]:
            return float(mean_absolute_error(y_true, y_pred))
        elif m in ["rmse", "neg_root_mean_squared_error"]:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif m in ["mse", "neg_mean_squared_error"]:
            return float(mean_squared_error(y_true, y_pred))
        elif m in ["r2", "r²"]:
            return float(r2_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported evaluation metric: '{metric_name}'.")

    def evaluate(self, cv: Optional[int] = None) -> "AutoML":
        """
        Evaluate every supported baseline model using leak-free cross-validation.

        Classification: StratifiedKFold
        Regression: KFold

        In every fold:
          training fold
          → fit preprocessing (ColumnTransformer)
          → transform training fold
          → transform validation fold
          → train model
          → evaluate

        Parameters
        ----------
        cv : int, optional
            Number of folds (cv >= 2). If not provided, uses the instance's cv parameter.

        Returns
        -------
        AutoML
            Evaluated AutoML instance.
        """
        k_folds = self._cv if cv is None else int(cv)
        if isinstance(k_folds, bool) or not isinstance(k_folds, int) or k_folds < 2:
            raise ValueError(f"cv must be an integer >= 2, got {k_folds}.")
        if len(self._df) < k_folds:
            raise ValueError(f"Dataset has only {len(self._df)} rows, which is less than cv={k_folds}.")

        task = self.task
        target_series = self._df[self._target]

        if task == "classification":
            min_class_count = int(target_series.value_counts().min())
            if min_class_count < k_folds:
                raise ValueError(
                    f"Cannot perform {k_folds}-fold cross-validation: the least populated class "
                    f"has only {min_class_count} sample(s), which is less than cv={k_folds}."
                )
            splitter = StratifiedKFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=self._random_state,
            )
            models = {
                "Logistic Regression": LogisticRegression(
                    random_state=self._random_state,
                    max_iter=1000,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=self._random_state,
                ),
                "Random Forest": RandomForestClassifier(
                    random_state=self._random_state,
                ),
            }
        else:
            splitter = KFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=self._random_state,
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=self._random_state,
                ),
                "Random Forest": RandomForestRegressor(
                    random_state=self._random_state,
                ),
            }

        X = self._df.drop(columns=[self._target]).copy()
        y = target_series.copy()

        self._cv_scores = {name: [] for name in models.keys()}

        # Evaluate each model across all folds
        for name, base_model in models.items():
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
                X_train_f = X.iloc[train_idx].copy()
                X_val_f = X.iloc[val_idx].copy()
                y_train_f = y.iloc[train_idx].copy()
                y_val_f = y.iloc[val_idx].copy()

                # Clean categorical missing representations
                for c in self._cat_cols:
                    X_train_f[c] = X_train_f[c].replace({None: np.nan})
                    X_val_f[c] = X_val_f[c].replace({None: np.nan})

                preprocessor = self._build_preprocessor()

                # 1. Fit preprocessing strictly on training fold only (prevent leakage)
                X_train_trans = preprocessor.fit_transform(X_train_f)
                # 2. Transform validation fold using fitted preprocessor
                X_val_trans = preprocessor.transform(X_val_f)

                # 3. Train model on training fold
                fold_model = clone(base_model)
                fold_model.fit(X_train_trans, y_train_f)

                # 4. Predict on validation fold
                y_val_pred = fold_model.predict(X_val_trans)

                # 5. Evaluate fold score
                score = self._compute_metric_score(
                    y_val_f, y_val_pred, self._primary_metric
                )
                self._cv_scores[name].append(score)

        # Build results DataFrame
        records = []
        for model_name in models.keys():
            scores = self._cv_scores[model_name]
            mean_val = round(float(np.mean(scores)), 4)
            std_val = round(float(np.std(scores)), 4)
            row: dict[str, Any] = {
                "Model": model_name,
                f"Mean {self._primary_metric_label}": mean_val,
                "Std": std_val,
            }
            for i, score_val in enumerate(scores, start=1):
                row[f"Fold {i}"] = round(float(score_val), 4)
            records.append(row)

        self._cv_results_df = ResultsDataFrame(records)
        self._is_evaluated = True
        return self

    # --------------------------------------------------------------------------
    # Controlled Hyperparameter Tuning (tune)
    # --------------------------------------------------------------------------

    def tune(self, n_iter: Optional[int] = None) -> "AutoML":
        """
        Execute controlled hyperparameter tuning using sklearn RandomizedSearchCV.
        Ensures zero data leakage by nesting ColumnTransformer preprocessing inside
        the Pipeline for every fold.

        Parameters
        ----------
        n_iter : int, optional
            Number of parameter combinations to sample (n_iter >= 1).
            If None, uses instance's n_iter configuration.

        Returns
        -------
        AutoML
            Tuned AutoML instance.
        """
        iter_count = self._n_iter if n_iter is None else int(n_iter)
        if isinstance(iter_count, bool) or not isinstance(iter_count, int) or iter_count < 1:
            raise ValueError(f"n_iter must be an integer >= 1, got {iter_count}.")

        task = self.task
        X = self._df.drop(columns=[self._target]).copy()
        y = self._df[self._target].copy()

        # Sanitize categorical missing values upfront in X
        for c in self._cat_cols:
            X[c] = X[c].replace({None: np.nan})

        # Separate held-out test set
        stratify = (
            y
            if (task == "classification" and int(y.value_counts().min()) >= 2)
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=stratify,
        )

        self._X_train_tune = X_train
        self._X_test_heldout = X_test
        self._y_train_tune = y_train
        self._y_test_heldout = y_test

        # Set up CV splitter for RandomizedSearchCV
        k_folds = self._cv
        if len(X_train) < k_folds:
            k_folds = max(2, len(X_train))

        if task == "classification":
            min_class_train = int(y_train.value_counts().min())
            if min_class_train < k_folds:
                k_folds = max(2, min_class_train)
            cv_splitter = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=self._random_state
            )
            # Define search models and compact parameter spaces
            models_and_params: dict[str, tuple[Any, dict[str, list[Any]]]] = {
                "Logistic Regression": (
                    LogisticRegression(
                        random_state=self._random_state, max_iter=1000
                    ),
                    {
                        "model__C": [0.01, 0.1, 1.0, 10.0],
                        "model__solver": ["lbfgs", "liblinear"],
                    },
                ),
                "Decision Tree": (
                    DecisionTreeClassifier(random_state=self._random_state),
                    {
                        "model__max_depth": [3, 5, 10, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4],
                    },
                ),
                "Random Forest": (
                    RandomForestClassifier(random_state=self._random_state),
                    {
                        "model__n_estimators": [20, 50, 100],
                        "model__max_depth": [3, 5, 10, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4],
                        "model__max_features": ["sqrt", "log2", None],
                    },
                ),
            }

            # Map primary metric to sklearn scoring string
            if self._primary_metric in ["f1", "f1_weighted"]:
                unique_train = np.unique(y_train)
                if len(unique_train) == 2 and set(unique_train).issubset({0, 1, True, False}):
                    sklearn_scoring = "f1"
                else:
                    sklearn_scoring = "f1_weighted"
            elif self._primary_metric in ["precision", "precision_weighted"]:
                sklearn_scoring = "precision_weighted"
            elif self._primary_metric in ["recall", "recall_weighted"]:
                sklearn_scoring = "recall_weighted"
            else:
                sklearn_scoring = self._primary_metric

        else:
            cv_splitter = KFold(
                n_splits=k_folds, shuffle=True, random_state=self._random_state
            )
            models_and_params = {
                "Linear Regression": (
                    LinearRegression(),
                    {"model__fit_intercept": [True]},
                ),
                "Decision Tree": (
                    DecisionTreeRegressor(random_state=self._random_state),
                    {
                        "model__max_depth": [3, 5, 10, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4],
                    },
                ),
                "Random Forest": (
                    RandomForestRegressor(random_state=self._random_state),
                    {
                        "model__n_estimators": [20, 50, 100],
                        "model__max_depth": [3, 5, 10, None],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4],
                        "model__max_features": ["sqrt", "log2", None],
                    },
                ),
            }

            if self._primary_metric in ["r2", "r²"]:
                sklearn_scoring = "r2"
            elif self._primary_metric in ["mae", "neg_mean_absolute_error"]:
                sklearn_scoring = "neg_mean_absolute_error"
            elif self._primary_metric in ["rmse", "neg_root_mean_squared_error"]:
                sklearn_scoring = "neg_root_mean_squared_error"
            elif self._primary_metric in ["mse", "neg_mean_squared_error"]:
                sklearn_scoring = "neg_mean_squared_error"
            else:
                sklearn_scoring = self._primary_metric

        self._tuned_searches = {}

        # Run RandomizedSearchCV for each model
        for name, (base_model, param_dist) in models_and_params.items():
            preprocessor = self._build_preprocessor()
            pipe = Pipeline([("prep", preprocessor), ("model", base_model)])

            # Calculate total parameter combinations to avoid sampling more than available
            comb_count = 1
            for p_vals in param_dist.values():
                comb_count *= len(p_vals)
            actual_n_iter = min(iter_count, comb_count)

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=actual_n_iter,
                cv=cv_splitter,
                scoring=sklearn_scoring,
                random_state=self._random_state,
                refit=True,
                n_jobs=1,
            )
            search.fit(X_train, y_train)

            raw_score = float(search.best_score_)
            # Convert negative sklearn metrics to positive values for reporting
            display_score = abs(raw_score) if "neg_" in sklearn_scoring else raw_score

            # Clean parameter names (strip 'model__' prefix for presentation)
            clean_params = {
                k.replace("model__", ""): v for k, v in search.best_params_.items()
            }

            self._tuned_searches[name] = {
                "search": search,
                "best_score": round(display_score, 4),
                "raw_score": raw_score,
                "best_params": clean_params,
                "n_iter": actual_n_iter,
            }

        # Identify top tuned model
        # For error metrics, lowest is best; for others, highest is best
        lower_is_better = self._primary_metric in [
            "mae",
            "rmse",
            "mse",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
        sorted_models = sorted(
            self._tuned_searches.items(),
            key=lambda item: item[1]["best_score"],
            reverse=not lower_is_better,
        )

        self._top_tuned_name = sorted_models[0][0]
        self._top_tuned_pipeline = sorted_models[0][1]["search"].best_estimator_
        self._pipeline = self._top_tuned_pipeline
        self._best_score = float(sorted_models[0][1]["best_score"])
        self._training_timestamp = datetime.now(timezone.utc).isoformat()

        # Build tuning results DataFrame
        tuning_records = []
        for name, data in sorted_models:
            tuning_records.append({
                "Model": name,
                f"Best Score ({self._primary_metric_label})": data["best_score"],
                "Best Parameters": data["best_params"],
                "Iterations": data["n_iter"],
            })

        self._tuning_results_df = ResultsDataFrame(tuning_records)
        self._is_tuned = True
        return self

    def tuning_results(self) -> pd.DataFrame:
        """
        Return summary table of hyperparameter tuning results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing model, best score, best parameters, and number of iterations.

        Raises
        ------
        RuntimeError
            If called before tune().
        """
        if not self._is_tuned or self._tuning_results_df is None:
            raise RuntimeError(
                "tuning_results() cannot be called before tune(). Please call automl.tune() first."
            )
        return self._tuning_results_df

    def best_models(self) -> dict[str, Any]:
        """
        Return the tuned models ordered by highest cross-validation score for the selected metric.

        Returns
        -------
        dict[str, Pipeline]
            Mapping from model name to best fitted Pipeline estimator.

        Raises
        ------
        RuntimeError
            If called before tune().
        """
        if not self._is_tuned or not self._tuned_searches:
            raise RuntimeError(
                "best_models() cannot be called before tune(). Please call automl.tune() first."
            )

        lower_is_better = self._primary_metric in [
            "mae",
            "rmse",
            "mse",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ]
        sorted_models = sorted(
            self._tuned_searches.items(),
            key=lambda item: item[1]["best_score"],
            reverse=not lower_is_better,
        )
        return {name: data["search"].best_estimator_ for name, data in sorted_models}

    # --------------------------------------------------------------------------
    # Prediction, Persistence & Evaluation
    # --------------------------------------------------------------------------

    def _validate_and_prepare_input(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Validate input features for prediction, checking for required columns,
        retaining correct order, and handling missing categorical values.
        Never modifies the caller's input DataFrame in place.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame for prediction is empty.")

            if self._feature_names:
                missing_cols = [c for c in self._feature_names if c not in X.columns]
                if missing_cols:
                    raise ValueError(
                        f"Input data is missing required feature column(s): {missing_cols}."
                    )
                # Keep strictly the feature columns in the expected order
                X_clean = X[self._feature_names].copy()
            else:
                X_clean = X.copy()

            # Handle None in categorical columns so sklearn SimpleImputer won't fail
            for c in self._cat_cols:
                if c in X_clean.columns:
                    X_clean[c] = X_clean[c].replace({None: np.nan})

            return X_clean
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input array for prediction is empty.")
            if X.ndim == 1:
                return X.reshape(1, -1)
            return X.copy()
        else:
            raise TypeError(
                f"predict() expects a pandas DataFrame, Series, or numpy array, got {type(X).__name__}."
            )

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Generate predictions using the active or loaded trained pipeline.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            Input features to generate predictions for. Supports full DataFrames,
            single-row DataFrames, or single-row Series.

        Returns
        -------
        np.ndarray
            Array of predicted values.

        Raises
        ------
        RuntimeError
            If neither tune(), fit(), nor load_model() has been called.
        ValueError
            If required feature columns are missing or input is empty.
        """
        if not self._is_tuned and not self._is_fitted and not self._is_loaded:
            raise RuntimeError(
                "predict() cannot be called before tuning, fitting, or loading a model. "
                "Please call automl.tune(), automl.fit(), or automl.load_model() first."
            )

        X_clean = self._validate_and_prepare_input(X)

        pipeline = self._top_tuned_pipeline or self._pipeline
        if pipeline is not None:
            return pipeline.predict(X_clean)
        elif self._is_fitted:
            if self._prep is None:
                raise RuntimeError("Preprocessing pipeline not found. Please call fit() or tune().")
            X_trans = self._prep.transform(X_clean)
            top_model_name = list(self._fitted_models.keys())[0]
            return self._fitted_models[top_model_name].predict(X_trans)
        else:
            raise RuntimeError("No model pipeline available for prediction.")

    def predict_proba(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """
        Generate predicted class probabilities for classification models.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or np.ndarray
            Input features to generate probability predictions for.

        Returns
        -------
        np.ndarray
            Array of predicted class probabilities of shape (n_samples, n_classes).

        Raises
        ------
        RuntimeError
            If neither tune(), fit(), nor load_model() has been called.
        ValueError
            If task is regression, required columns are missing, or input is empty.
        AttributeError
            If the trained model does not support probability estimation.
        """
        if not self._is_tuned and not self._is_fitted and not self._is_loaded:
            raise RuntimeError(
                "predict_proba() cannot be called before tuning, fitting, or loading a model. "
                "Please call automl.tune(), automl.fit(), or automl.load_model() first."
            )

        if self._task == "regression":
            raise ValueError(
                "predict_proba() is only supported for classification tasks, "
                "but the current task is regression."
            )

        X_clean = self._validate_and_prepare_input(X)

        pipeline = self._top_tuned_pipeline or self._pipeline
        if pipeline is not None:
            final_model = pipeline.named_steps.get("model", None)
            if final_model is not None and not hasattr(final_model, "predict_proba"):
                raise AttributeError(
                    f"The active model '{type(final_model).__name__}' does not support predict_proba."
                )
            if not hasattr(pipeline, "predict_proba"):
                raise AttributeError(
                    f"The active model pipeline does not support predict_proba."
                )
            return pipeline.predict_proba(X_clean)
        elif self._is_fitted:
            top_model_name = list(self._fitted_models.keys())[0]
            model = self._fitted_models[top_model_name]
            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    f"Model '{top_model_name}' does not support predict_proba."
                )
            if self._prep is None:
                raise RuntimeError("Preprocessing pipeline not found.")
            X_trans = self._prep.transform(X_clean)
            return model.predict_proba(X_trans)
        else:
            raise RuntimeError("No model pipeline available for probability prediction.")

    def save_model(self, filepath: Union[str, Path]) -> str:
        """
        Save the complete trained pipeline, including preprocessing, selected model,
        and required metadata using joblib.

        Parameters
        ----------
        filepath : str or Path
            Path where the model bundle should be saved (e.g. 'model.pkl').

        Returns
        -------
        str
            The file path where the model was successfully saved.

        Raises
        ------
        RuntimeError
            If called before tune(), fit(), or load_model().
        ValueError
            If filepath is empty or points to a directory.
        """
        if not self._is_tuned and not self._is_fitted and not self._is_loaded:
            raise RuntimeError(
                "save_model() requires a fitted, tuned, or loaded model. "
                "Please call automl.tune() or automl.fit() first."
            )

        if not filepath:
            raise ValueError("filepath must be a non-empty string or Path.")

        target_path = Path(filepath)
        if target_path.is_dir():
            raise ValueError(f"filepath '{filepath}' is a directory. Please specify a file path.")

        # Ensure parent directory exists
        if target_path.parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_to_save = self._top_tuned_pipeline or self._pipeline
        if pipeline_to_save is None and self._is_fitted:
            if self._prep is None:
                raise RuntimeError("Preprocessing pipeline not found.")
            best_model_name = list(self._fitted_models.keys())[0]
            best_model = self._fitted_models[best_model_name]
            pipeline_to_save = Pipeline([
                ("prep", self._prep.get_pipeline()),
                ("model", best_model),
            ])
            self._top_tuned_pipeline = pipeline_to_save
            self._top_tuned_name = best_model_name

        if pipeline_to_save is None:
            raise RuntimeError("No model pipeline available to save.")

        model_step = pipeline_to_save.named_steps.get("model")
        model_class_name = (
            type(model_step).__name__
            if model_step is not None
            else str(self._top_tuned_name or "Model")
        )

        bundle = {
            "tabulixml_signature": "TABULIXML_MODEL_BUNDLE",
            "version": TABULIXML_VERSION,
            "task": self._task,
            "target": self._target,
            "model": model_class_name,
            "model_name": self._top_tuned_name or model_class_name,
            "feature_names": list(self._feature_names) if self._feature_names else [],
            "cat_cols": list(self._cat_cols) if self._cat_cols else [],
            "num_cols": list(self._num_cols) if self._num_cols else [],
            "training_timestamp": (
                self._training_timestamp
                if self._training_timestamp is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            "metric": getattr(self, "_primary_metric", None),
            "best_score": getattr(self, "_best_score", None),
            "pipeline": pipeline_to_save,
        }

        joblib.dump(bundle, str(target_path))
        return str(target_path)

    @_ClassOrInstanceMethod
    def load_model(cls_or_self: Any, filepath: Union[str, Path]) -> "AutoML":
        """
        Load a previously saved TabulixML model.

        The loaded object is fully configured and ready for prediction without retraining.
        Can be called as an instance method (e.g. `automl.load_model(...)`) or as a class
        method (e.g. `automl = AutoML.load_model(...)`).

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model file.

        Returns
        -------
        AutoML
            The populated AutoML instance.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        ValueError
            If the file is corrupt or not a valid TabulixML model bundle.
        """
        if isinstance(cls_or_self, type):
            instance = cls_or_self.__new__(cls_or_self)
            instance._init_empty()
            instance._load_bundle(filepath)
            return instance
        else:
            cls_or_self._load_bundle(filepath)
            return cls_or_self

    def _load_bundle(self, filepath: Union[str, Path]) -> None:
        """Internal helper to deserialize and restore model state from a joblib file."""
        if not filepath:
            raise ValueError("filepath must be a non-empty string or Path.")

        target_path = Path(filepath)
        if not target_path.is_file():
            raise FileNotFoundError(f"Model file not found: '{filepath}'.")

        try:
            bundle = joblib.load(str(target_path))
        except Exception as e:
            raise ValueError(f"Failed to load model file '{filepath}': {e}") from e

        if not isinstance(bundle, dict):
            raise ValueError(
                f"File '{filepath}' is not a valid TabulixML model bundle "
                f"(expected dictionary, got {type(bundle).__name__})."
            )

        if "pipeline" not in bundle or "task" not in bundle:
            raise ValueError(
                f"File '{filepath}' is missing essential TabulixML bundle components "
                f"('pipeline' or 'task')."
            )

        self._pipeline = bundle["pipeline"]
        self._top_tuned_pipeline = bundle["pipeline"]
        self._task = bundle.get("task")
        self._target = bundle.get("target")
        self._top_tuned_name = bundle.get("model", bundle.get("model_name", "TrainedModel"))
        self._feature_names = bundle.get("feature_names", [])
        self._feature_cols = list(self._feature_names)
        self._cat_cols = bundle.get("cat_cols", [])
        self._num_cols = bundle.get("num_cols", [])
        self._training_timestamp = bundle.get("training_timestamp")
        self._primary_metric = bundle.get("metric", "auto")
        self._best_score = bundle.get("best_score")
        self._metadata = bundle
        self._is_loaded = True
        self._is_tuned = True
        self._is_empty = False

    def model_info(self) -> dict[str, Any]:
        """
        Return metadata and configuration details of the active/loaded model.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model metadata:
            - 'task': str ('classification' or 'regression')
            - 'target': str
            - 'model': str (e.g. 'RandomForestClassifier')
            - 'version': str (TabulixML version, e.g. '1.0.0')
            - 'feature_names': list[str]
            - 'training_timestamp': str or None
            - 'metric': str or None
            - 'best_score': float or None

        Raises
        ------
        RuntimeError
            If no model has been fitted, tuned, or loaded yet.
        """
        if not self._is_tuned and not self._is_fitted and not self._is_loaded:
            raise RuntimeError(
                "model_info() requires a fitted, tuned, or loaded model. "
                "Please call automl.tune(), automl.fit(), or automl.load_model() first."
            )

        pipeline = self._top_tuned_pipeline or self._pipeline
        if pipeline is not None and "model" in pipeline.named_steps:
            model_class = type(pipeline.named_steps["model"]).__name__
        elif self._is_fitted and self._fitted_models:
            first_model = list(self._fitted_models.values())[0]
            model_class = type(first_model).__name__
        else:
            model_class = str(self._top_tuned_name or "Unknown")

        info: dict[str, Any] = {
            "task": self._task,
            "target": self._target,
            "model": model_class,
            "version": TABULIXML_VERSION,
            "feature_names": list(self._feature_names) if self._feature_names else [],
            "training_timestamp": self._training_timestamp,
        }
        if getattr(self, "_primary_metric", None) is not None:
            info["metric"] = self._primary_metric
        if getattr(self, "_best_score", None) is not None:
            info["best_score"] = round(float(self._best_score), 4)

        return info

    def evaluate_tuned(
        self,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        print_report: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate the tuned model with the highest cross-validation score on the held-out test set
        (or custom test data).

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray, optional
            Test feature matrix. If None, uses held-out partition from tune().
        y_test : pd.Series or np.ndarray, optional
            Test ground-truth target. If None, uses held-out partition from tune().
        print_report : bool, default True
            Whether to print the evaluation summary to stdout.

        Returns
        -------
        pd.DataFrame
            DataFrame containing held-out test evaluation metrics and scores.

        Raises
        ------
        RuntimeError
            If called before tune() or load_model() without supplying (X_test, y_test).
        """
        if (
            not self._is_tuned
            and not self._is_loaded
            and self._top_tuned_pipeline is None
        ):
            raise RuntimeError(
                "evaluate_tuned() cannot be called before tune() or loading a model. "
                "Please call automl.tune() first."
            )

        if X_test is not None and y_test is not None:
            eval_X = self._validate_and_prepare_input(X_test)
            eval_y = np.asarray(y_test)
        else:
            if self._X_test_heldout is None or self._y_test_heldout is None:
                raise RuntimeError(
                    "evaluate_tuned() requires held-out test data from tune(), or pass (X_test, y_test) explicitly."
                )
            eval_X = self._X_test_heldout
            eval_y = self._y_test_heldout

        pipeline = self._top_tuned_pipeline or self._pipeline
        if pipeline is None:
            raise RuntimeError("No model pipeline available for evaluation.")

        y_test_pred = pipeline.predict(eval_X)
        y_test_true = eval_y

        if self.task == "classification":
            acc = float(accuracy_score(y_test_true, y_test_pred))
            unique_classes = np.unique(y_test_true)
            if len(unique_classes) == 2 and set(unique_classes).issubset({0, 1, True, False}):
                prec = float(precision_score(y_test_true, y_test_pred, zero_division=0))
                rec = float(recall_score(y_test_true, y_test_pred, zero_division=0))
                f1 = float(f1_score(y_test_true, y_test_pred, zero_division=0))
            else:
                prec = float(
                    precision_score(y_test_true, y_test_pred, average="weighted", zero_division=0)
                )
                rec = float(
                    recall_score(y_test_true, y_test_pred, average="weighted", zero_division=0)
                )
                f1 = float(f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0))

            records = [
                {"Metric": "accuracy", "Score": round(acc, 4)},
                {"Metric": "precision", "Score": round(prec, 4)},
                {"Metric": "recall", "Score": round(rec, 4)},
                {"Metric": "F1", "Score": round(f1, 4)},
            ]
        else:
            mae = float(mean_absolute_error(y_test_true, y_test_pred))
            mse = float(mean_squared_error(y_test_true, y_test_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test_true, y_test_pred))
            records = [
                {"Metric": "MAE", "Score": round(mae, 4)},
                {"Metric": "RMSE", "Score": round(rmse, 4)},
                {"Metric": "R²", "Score": round(r2, 4)},
            ]

        results_df = ResultsDataFrame(records)

        if print_report:
            print("=" * 60)
            print("TabulixML AutoML Held-Out Test Evaluation")
            print(f"Tuned Model with highest CV score : {self._top_tuned_name}")
            print("-" * 60)
            print(results_df.to_string(index=False))
            print("=" * 60)

        return results_df

    # --------------------------------------------------------------------------
    # Results & Model Comparison
    # --------------------------------------------------------------------------

    def results(self, source: str = "auto") -> pd.DataFrame:
        """
        Return evaluation results table.

        Parameters
        ----------
        source : {'auto', 'evaluate', 'fit', 'tune'}, default 'auto'
            Which results to return:
              - 'auto'     : Returns cross-validation or tuning results if executed,
                             otherwise returns fit() results.
              - 'evaluate' : Cross-validation results table.
              - 'tune'     : Hyperparameter tuning results table.
              - 'fit'      : Single train/test split results table.

        Returns
        -------
        pd.DataFrame
            DataFrame containing model evaluation results.

        Raises
        ------
        RuntimeError
            If neither fit(), evaluate(), nor tune() has been executed yet.
        """
        if source == "auto":
            if self._is_tuned and self._tuning_results_df is not None:
                return self._tuning_results_df
            if self._is_evaluated and self._cv_results_df is not None:
                return self._cv_results_df
            if self._is_fitted and self._results_df is not None:
                return self._results_df
            raise RuntimeError(
                "results() cannot be called before fit(), evaluate(), or tune(). "
                "Please call automl.fit(), automl.evaluate(), or automl.tune() first."
            )
        elif source in ["evaluate", "cv"]:
            if not self._is_evaluated or self._cv_results_df is None:
                raise RuntimeError(
                    "Cross-validation results not found. Please call automl.evaluate() first."
                )
            return self._cv_results_df
        elif source == "tune":
            if not self._is_tuned or self._tuning_results_df is None:
                raise RuntimeError(
                    "Tuning results not found. Please call automl.tune() first."
                )
            return self._tuning_results_df
        elif source == "fit":
            if not self._is_fitted or self._results_df is None:
                raise RuntimeError("Fit results not found. Please call automl.fit() first.")
            return self._results_df
        else:
            raise ValueError(f"Invalid source '{source}'. Expected 'auto', 'evaluate', 'tune', or 'fit'.")

    def compare(self, print_table: bool = True) -> pd.DataFrame:
        """
        Display a clean comparison table sorted by the primary cross-validation metric.
        Labels the highest-scoring model for the selected metric.

        Parameters
        ----------
        print_table : bool, default True
            Whether to print the formatted comparison table to stdout.

        Returns
        -------
        pd.DataFrame
            Sorted results DataFrame.

        Raises
        ------
        RuntimeError
            If evaluate() has not been called yet.
        """
        if not self._is_evaluated or self._cv_results_df is None:
            raise RuntimeError(
                "compare() cannot be called before evaluate(). Please call automl.evaluate() first."
            )

        mean_col = f"Mean {self._primary_metric_label}"

        lower_is_better = self._primary_metric in [
            "mae",
            "rmse",
            "mse",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
            "neg_mean_squared_error",
        ]
        sorted_df = self._cv_results_df.sort_values(
            by=mean_col, ascending=lower_is_better
        ).reset_index(drop=True)
        sorted_results = ResultsDataFrame(sorted_df)

        top_model = sorted_results.iloc[0]["Model"]
        top_mean = sorted_results.iloc[0][mean_col]
        top_std = sorted_results.iloc[0]["Std"]

        sep = "=" * 76
        sub_sep = "-" * 76
        lines = [
            sep,
            "TabulixML AutoML Model Comparison",
            f"Task: {self.task} | Metric: {self._primary_metric_label} | Cross-Validation: {self._cv} folds",
            sep,
            sorted_results.to_string(index=False),
            sub_sep,
            f"Highest-scoring model for {self._primary_metric_label}: {top_model} (Mean: {top_mean:.4f}, Std: {top_std:.4f})",
            sep,
        ]
        comparison_text = "\n".join(lines)

        if print_table:
            print(comparison_text)

        return sorted_results

    def report(self, print_report: bool = True) -> str:
        """
        Print and return a readable summary of:
          - detected task
          - dataset size
          - models tested
          - metrics
          - training / tuning status

        Parameters
        ----------
        print_report : bool, default True
            Whether to print the report to stdout.

        Returns
        -------
        str
            The formatted report text.
        """
        task = self.task
        status_parts = []
        if self._is_loaded:
            status_parts.append(f"loaded from file (model: {self._top_tuned_name})")
        if self._is_fitted:
            status_parts.append("fit() completed")
        if self._is_evaluated:
            status_parts.append(f"evaluate() completed ({self._cv}-fold CV)")
        if self._is_tuned:
            status_parts.append(f"tune() completed (top model: {self._top_tuned_name})")
        if not status_parts:
            status_parts.append("not fitted, evaluated, or tuned yet")
        status = ", ".join(status_parts)

        n_samples = len(self._df) if self._df is not None else "N/A (loaded model)"
        n_features = (len(self._df.columns) - 1) if self._df is not None else len(self._feature_names)

        models_tested = (
            ["Logistic Regression", "Decision Tree", "Random Forest"]
            if task == "classification"
            else ["Linear Regression", "Decision Tree", "Random Forest"]
        )

        lines = [
            "=" * 60,
            "TabulixML AutoML Report",
            "=" * 60,
            f"Detected Task   : {task}",
            f"Execution Status: {status}",
            f"Dataset Size    : {n_samples} rows, {n_features} features",
            f"Target Column   : {self._target}",
            f"Models Tested   : {', '.join(models_tested)}",
        ]

        if self._is_tuned and self._tuning_results_df is not None:
            lines.append("")
            lines.append(f"Hyperparameter Tuning Results (Metric: {self._primary_metric_label}):")
            lines.append("-" * 60)
            lines.append(self._tuning_results_df.to_string(index=False))
            lines.append("-" * 60)

        if self._is_evaluated and self._cv_results_df is not None:
            lines.append("")
            lines.append(f"Cross-Validation Summary ({self._cv} folds, Metric: {self._primary_metric_label}):")
            lines.append("-" * 60)
            lines.append(self._cv_results_df.to_string(index=False))
            lines.append("-" * 60)

        if self._is_fitted and self._scores:
            lines.append("")
            lines.append("Single Train/Test Split Evaluation Results:")
            lines.append("-" * 60)
            lines.append(f"{'Model':<24} {'Metric':<16} {'Score':<10}")
            lines.append("-" * 60)
            for model_name, metrics in self._scores.items():
                for metric_name, score in metrics.items():
                    lines.append(f"{model_name:<24} {metric_name:<16} {score:<10.4f}")
            lines.append("-" * 60)

        if not self._is_fitted and not self._is_evaluated and not self._is_tuned:
            lines.append("")
            lines.append("Note: Call automl.tune(), automl.evaluate(), or automl.fit() to start.")

        lines.append("=" * 60)
        report_text = "\n".join(lines)

        if print_report:
            print(report_text)

        return report_text

    def get_models(self) -> dict[str, Any]:
        """
        Return dictionary of trained scikit-learn model objects from fit().

        Returns
        -------
        dict[str, Any]
            Mapping of model names to fitted model instances.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("AutoML has not been fitted yet. Please call fit() first.")
        return dict(self._fitted_models)

    def get_prep(self) -> AutoPrep:
        """
        Return the fitted AutoPrep instance used internally during fit().

        Returns
        -------
        AutoPrep
            The fitted AutoPrep instance.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self._is_fitted or self._prep is None:
            raise RuntimeError("AutoML has not been fitted yet. Please call fit() first.")
        return self._prep
