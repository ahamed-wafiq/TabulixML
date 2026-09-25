"""
src/datacraft/prep.py -- AutoPrep Module for DataCraft

Provides leak-free, reproducible machine learning preprocessing with scikit-learn:
  - Column type detection: separates numerical, categorical, datetime, and target.
  - inspect(): Shows feature columns, target, type breakdown, missing, and unique counts.
  - preview(): Displays proposed preprocessing transformations upfront.
  - split(): Splits dataset into train and test sets using scikit-learn train_test_split.
  - prepare(): Splits data, fits preprocessing ONLY on training data, transforms train & test,
               and returns (X_train, X_test, y_train, y_test).
  - get_pipeline(): Returns the fitted sklearn ColumnTransformer pipeline.
  - transform(): Transforms new/unseen data using the fitted pipeline without refitting.
  - get_feature_names(): Returns preserved feature names after imputation, scaling, and one-hot encoding.

Guarantees:
  - Strict immutability: never modifies the caller's DataFrame in place.
  - Zero target leakage: target is isolated upfront; preprocessing is fitted ONLY on training data.
  - Handles unseen/unknown categorical values gracefully via OneHotEncoder(handle_unknown='ignore').
  - Handles datasets with only numerical or only categorical features safely.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datacraft.cleaner import AutoClean


class AutoPrep:
    """
    Automated ML Preprocessing for tabular datasets using scikit-learn.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to preprocess. Must be non-empty.
    target : str, optional
        Target column name to separate from feature preprocessing.
    test_size : float, default 0.2
        Fraction of the dataset to allocate to the test split (0.0 < test_size < 1.0).
    random_state : int or None, default 42
        Seed used by random number generator for reproducible train/test splits.
    scale : bool, default True
        Whether to standardize numerical features using StandardScaler.
    encode : bool, default True
        Whether to one-hot encode categorical features using OneHotEncoder.

    Examples
    --------
    >>> import pandas as pd
    >>> from datacraft import AutoPrep
    >>> df = pd.DataFrame({
    ...     "age": [25.0, 30.0, None, 45.0, 29.0],
    ...     "city": ["Paris", "London", "Paris", "Berlin", "London"],
    ...     "churn": [0, 1, 0, 1, 0]
    ... })
    >>> prep = AutoPrep(df, target="churn", test_size=0.2, random_state=42)
    >>> prep.inspect()
    >>> prep.preview()
    >>> X_train, X_test, y_train, y_test = prep.prepare()
    >>> pipeline = prep.get_pipeline()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        test_size: float = 0.2,
        random_state: Optional[int] = 42,
        scale: bool = True,
        encode: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"AutoPrep expects a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        if target is not None:
            if not isinstance(target, str):
                raise TypeError(f"Target column must be a string, got {type(target).__name__}.")
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame.")
            if df[target].isna().all():
                raise ValueError(f"Target column '{target}' contains only missing values.")

        if not isinstance(test_size, (int, float)) or not (0.0 < float(test_size) < 1.0):
            raise ValueError(f"test_size must be a float between 0.0 and 1.0 (exclusive), got {test_size}.")

        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0:
                raise ValueError(f"random_state must be a non-negative integer or None, got {random_state}.")

        if not isinstance(scale, bool):
            raise TypeError(f"scale must be a boolean, got {type(scale).__name__}.")
        if not isinstance(encode, bool):
            raise TypeError(f"encode must be a boolean, got {type(encode).__name__}.")

        self._df = df
        self._target = target
        self._test_size = float(test_size)
        self._random_state = random_state
        self._scale = scale
        self._encode = encode

        # Detect column types using AutoClean detection logic
        raw_types = AutoClean._detect_col_types(df)

        self._col_types: dict[str, str] = {}
        for col, ctype in raw_types.items():
            if col == self._target:
                self._col_types[col] = "target"
            else:
                self._col_types[col] = ctype

        self._feature_cols = [c for c in df.columns if c != self._target]
        if self._target is not None and len(self._feature_cols) == 0:
            raise ValueError(
                "DataFrame must contain at least one feature column besides the target column."
            )
        self._num_cols = [c for c in self._feature_cols if self._col_types[c] == "numerical"]
        self._cat_cols = [c for c in self._feature_cols if self._col_types[c] == "categorical"]
        self._dt_cols = [c for c in self._feature_cols if self._col_types[c] == "datetime"]

        self._pipeline: Optional[ColumnTransformer] = None
        self._X_train_raw: Optional[pd.DataFrame] = None
        self._X_test_raw: Optional[pd.DataFrame] = None
        self._y_train_raw: Optional[pd.Series] = None
        self._y_test_raw: Optional[pd.Series] = None

    @property
    def target(self) -> Optional[str]:
        """Return the target column name."""
        return self._target

    @property
    def feature_columns(self) -> list[str]:
        """Return list of feature column names."""
        return list(self._feature_cols)

    @property
    def numerical_columns(self) -> list[str]:
        """Return list of numerical feature column names."""
        return list(self._num_cols)

    @property
    def categorical_columns(self) -> list[str]:
        """Return list of categorical feature column names."""
        return list(self._cat_cols)

    @property
    def datetime_columns(self) -> list[str]:
        """Return list of datetime feature column names."""
        return list(self._dt_cols)

    @property
    def col_types(self) -> dict[str, str]:
        """Return mapping of column names to detected data types."""
        return dict(self._col_types)

    @property
    def test_size(self) -> float:
        """Return configured test_size ratio."""
        return self._test_size

    @property
    def random_state(self) -> Optional[int]:
        """Return configured random_state seed."""
        return self._random_state

    @property
    def pipeline(self) -> Optional[ColumnTransformer]:
        """Return the fitted ColumnTransformer pipeline if prepare() was called."""
        return self._pipeline

    # ------------------------------------------------------------------
    # 1. inspect()
    # ------------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        """
        Inspect features, target, data types, missing counts, and unique value counts.

        Returns
        -------
        dict with:
          - 'feature_columns' / 'features': list of feature column names
          - 'target_column' / 'target': target column name (or None)
          - 'numerical_columns' / 'numerical': list of numerical feature columns
          - 'categorical_columns' / 'categorical': list of categorical feature columns
          - 'datetime_columns' / 'datetime': list of datetime feature columns
          - 'missing_values' / 'missing': dict of column -> missing count
          - 'unique_values' / 'unique': dict of column -> unique count
          - 'col_types': dict of column -> detected type
        """
        missing_dict = {col: int(self._df[col].isnull().sum()) for col in self._df.columns}
        unique_dict = {col: int(self._df[col].nunique(dropna=True)) for col in self._df.columns}

        return {
            "feature_columns": list(self._feature_cols),
            "features": list(self._feature_cols),
            "target_column": self._target,
            "target": self._target,
            "numerical_columns": list(self._num_cols),
            "numerical": list(self._num_cols),
            "categorical_columns": list(self._cat_cols),
            "categorical": list(self._cat_cols),
            "datetime_columns": list(self._dt_cols),
            "datetime": list(self._dt_cols),
            "missing_values": missing_dict,
            "missing": missing_dict,
            "unique_values": unique_dict,
            "unique": unique_dict,
            "col_types": dict(self._col_types),
        }

    # ------------------------------------------------------------------
    # 2. preview()
    # ------------------------------------------------------------------

    def preview(self, print_preview: bool = True) -> dict[str, Any]:
        """
        Preview proposed preprocessing pipeline steps and configuration.

        Parameters
        ----------
        print_preview : bool, default True
            Whether to print the human-readable preview to stdout.

        Returns
        -------
        dict : Summary of planned transformations for numerical, categorical,
               datetime, and target columns.
        """
        num_impute = "median (SimpleImputer)" if self._num_cols else "None (no numerical features)"
        num_scale = (
            "StandardScaler"
            if (self._num_cols and self._scale)
            else ("None (scale=False)" if self._num_cols else "None")
        )

        cat_impute = (
            "mode (SimpleImputer, strategy='most_frequent')"
            if self._cat_cols
            else "None (no categorical features)"
        )
        cat_encode = (
            "OneHotEncoder (handle_unknown='ignore')"
            if (self._cat_cols and self._encode)
            else ("None (encode=False)" if self._cat_cols else "None")
        )

        dt_action = (
            "requires feature extraction (not automatically transformed)"
            if self._dt_cols
            else "None (no datetime features)"
        )

        target_action = (
            f"'{self._target}' separated and excluded from feature preprocessing"
            if self._target
            else "None specified"
        )

        plan = {
            "target": {
                "column": self._target,
                "action": target_action,
            },
            "split": {
                "test_size": self._test_size,
                "random_state": self._random_state,
            },
            "numerical": {
                "columns": list(self._num_cols),
                "missing_imputation": num_impute,
                "scaling": num_scale,
            },
            "categorical": {
                "columns": list(self._cat_cols),
                "missing_imputation": cat_impute,
                "encoding": cat_encode,
            },
            "datetime": {
                "columns": list(self._dt_cols),
                "action": dt_action,
            },
        }

        sep = "=" * 64
        sub_sep = "-" * 64
        lines = [
            sep,
            "  DataCraft -- AutoPrep Preview",
            sep,
            f"  Target Column : {self._target} (kept separate from feature transformations)",
            f"  Train/Test    : test_size={self._test_size}, random_state={self._random_state}",
            f"  Total Features: {len(self._feature_cols)} (Numerical: {len(self._num_cols)}, Categorical: {len(self._cat_cols)}, Datetime: {len(self._dt_cols)})",
            "",
            sub_sep,
            f"  Numerical Features ({len(self._num_cols)}):",
            sub_sep,
            f"    * Columns            : {self._num_cols}",
            f"    * Missing Imputation : {num_impute}",
            f"    * Scaling            : {num_scale}",
            "",
            sub_sep,
            f"  Categorical Features ({len(self._cat_cols)}):",
            sub_sep,
            f"    * Columns            : {self._cat_cols}",
            f"    * Missing Imputation : {cat_impute}",
            f"    * Encoding           : {cat_encode}",
        ]

        if self._dt_cols:
            lines.extend([
                "",
                sub_sep,
                f"  Datetime Features ({len(self._dt_cols)}):",
                sub_sep,
                f"    * Columns: {self._dt_cols}",
                f"    * Action : {dt_action}",
            ])

        lines.extend([
            sep,
            "",
        ])

        if print_preview:
            print("\n".join(lines))

        return plan

    # ------------------------------------------------------------------
    # 3. split()
    # ------------------------------------------------------------------

    def split(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into train and test sets using scikit-learn train_test_split.

        Parameters
        ----------
        test_size : float, optional
            Override instance test_size if provided.
        random_state : int, optional
            Override instance random_state seed if provided.

        Returns
        -------
        X_train : pd.DataFrame
            Raw feature columns for training split.
        X_test : pd.DataFrame
            Raw feature columns for testing split.
        y_train : pd.Series
            Target values for training split.
        y_test : pd.Series
            Target values for testing split.
        """
        if self._target is None:
            raise ValueError("Cannot perform split(): target column must be specified.")

        if len(self._df) < 2:
            raise ValueError(
                f"Dataset has too few samples for train/test splitting "
                f"(found {len(self._df)} rows, minimum required is 2)."
            )

        t_size = self._test_size if test_size is None else float(test_size)
        if not (0.0 < t_size < 1.0):
            raise ValueError(f"test_size must be a float between 0.0 and 1.0 (exclusive), got {t_size}.")

        r_state = self._random_state if random_state is None else random_state
        if r_state is not None and (not isinstance(r_state, int) or r_state < 0):
            raise ValueError(f"random_state must be a non-negative integer or None, got {r_state}.")

        X_df = self._df[self._feature_cols].copy()
        y_series = self._df[self._target].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_df,
            y_series,
            test_size=t_size,
            random_state=r_state,
        )

        self._X_train_raw = X_train
        self._X_test_raw = X_test
        self._y_train_raw = y_train
        self._y_test_raw = y_test

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # 4. prepare()
    # ------------------------------------------------------------------

    def prepare(self) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Execute full ML preprocessing workflow:
          1. Split data into train and test sets.
          2. Build scikit-learn preprocessing pipeline.
          3. Fit the preprocessing pipeline ONLY on X_train (prevents target/test data leakage).
          4. Transform X_train using fitted pipeline.
          5. Transform X_test using fitted pipeline.

        Returns
        -------
        X_train : np.ndarray
            Transformed training feature matrix.
        X_test : np.ndarray
            Transformed testing feature matrix.
        y_train : pd.Series
            Target values for training split.
        y_test : pd.Series
            Target values for testing split.
        """
        if self._target is None:
            raise ValueError("prepare() cannot be called before a valid target column is provided.")

        # 1. Split data (or reuse cached split)
        X_train_df, X_test_df, y_train, y_test = self.split()

        # 2. Build preprocessing pipeline
        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if self._num_cols:
            num_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="median", missing_values=np.nan))
            ]
            if self._scale:
                num_steps.append(("scaler", StandardScaler()))
            num_pipe = Pipeline(steps=num_steps)
            transformers.append(("num", num_pipe, list(self._num_cols)))

        if self._cat_cols:
            cat_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan))
            ]
            if self._encode:
                cat_steps.append(
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                )
            cat_pipe = Pipeline(steps=cat_steps)
            transformers.append(("cat", cat_pipe, list(self._cat_cols)))

        if not transformers:
            raise ValueError("No transformable numerical or categorical feature columns found.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # 3. Clean object columns for training
        X_train_clean = X_train_df.copy()
        for col in self._cat_cols:
            X_train_clean[col] = X_train_clean[col].replace({None: np.nan})

        # 4. Fit ONLY on X_train and transform X_train
        preprocessor.fit(X_train_clean)
        self._pipeline = preprocessor
        X_train = preprocessor.transform(X_train_clean)

        # 5. Transform X_test (without refitting!)
        X_test_clean = X_test_df.copy()
        for col in self._cat_cols:
            X_test_clean[col] = X_test_clean[col].replace({None: np.nan})
        X_test = preprocessor.transform(X_test_clean)

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # 5. get_pipeline()
    # ------------------------------------------------------------------

    def get_pipeline(self) -> ColumnTransformer:
        """
        Return the fitted sklearn preprocessing pipeline.

        Returns
        -------
        ColumnTransformer : Fitted sklearn ColumnTransformer.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call prepare() first.")
        return self._pipeline

    # ------------------------------------------------------------------
    # 6. transform()
    # ------------------------------------------------------------------

    def transform(self, new_data: Union[pd.DataFrame, dict]) -> np.ndarray:
        """
        Transform new or unseen data using the already-fitted pipeline.
        Strictly preserves fitted state without refitting.

        Parameters
        ----------
        new_data : pd.DataFrame or dict
            Unseen feature data to transform.

        Returns
        -------
        np.ndarray : Transformed feature matrix.
        """
        if self._pipeline is None:
            raise RuntimeError("transform() cannot be called before prepare(). Call prepare() first.")

        if not isinstance(new_data, pd.DataFrame):
            try:
                new_data = pd.DataFrame(new_data)
            except Exception as e:
                raise TypeError(f"new_data must be a pandas DataFrame or dict, got {type(new_data).__name__}.") from e

        missing_cols = [c for c in self._feature_cols if c not in new_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns in new_data: {missing_cols}")

        df_clean = new_data[self._feature_cols].copy()
        for col in self._cat_cols:
            df_clean[col] = df_clean[col].replace({None: np.nan})

        return self._pipeline.transform(df_clean)

    # ------------------------------------------------------------------
    # 7. get_feature_names()
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """
        Return preserved feature names after imputation, scaling, and one-hot encoding.

        Returns
        -------
        list of str : Output feature names.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline has not been fitted yet. Call prepare() first.")
        return list(self._pipeline.get_feature_names_out())
