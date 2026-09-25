# TabulixML API Reference

Complete API reference for TabulixML v1.0.0.

```python
from tabulixml import AutoClean, AutoEDA, AutoPrep, AutoML
```

---

## 1. AutoClean

Automated tabular cleaning and data hygiene.

```python
cleaner = AutoClean(
    df: pd.DataFrame,
    missing_strategy: str = "auto",
    remove_duplicates: bool = True,
    normalize_names: bool = True,
    iqr_multiplier: float = 1.5,
    correlation_threshold: float = 0.90,
    target_column: Optional[str] = None
)
```

### Methods

- `inspect() -> Dict[str, Any]`: Returns summary of missing values, duplicates, column types, and warnings.
- `preview() -> pd.DataFrame`: Returns a preview of changes without applying them.
- `clean(approve: bool = True) -> pd.DataFrame`: Executes the cleaning pipeline and returns a cleaned DataFrame.
- `report() -> Dict[str, Any]`: Returns a structured dictionary of before/after metrics and operations applied.
- `history() -> List[Dict[str, Any]]`: Returns the ordered step-by-step history of transformations performed.
- `reset() -> None`: Resets the cleaner to its initial state.

---

## 2. AutoEDA

Read-only exploratory data analysis, statistics, and visualization.

```python
eda = AutoEDA(
    df: pd.DataFrame,
    correlation_threshold: float = 0.80
)
```

### Methods

- `inspect() -> Dict[str, Any]`: Returns basic shapes, column data types, missing counts, and duplicate row count.
- `summary() -> Dict[str, pd.DataFrame]`: Computes descriptive statistics for numerical and categorical features.
- `correlations(threshold: Optional[float] = None) -> Dict[str, Any]`: Generates Pearson correlation matrix and flags high-correlation pairs.
- `quality() -> Dict[str, Any]`: Evaluates missing rates, cardinality, duplicate rates, and outlier counts.
- `visualize(output_dir: Optional[str] = None, show: bool = False) -> List[str]`: Generates visualization plots (histograms, boxplots, correlation heatmap) and returns saved file paths.
- `save_report(filepath: str = "eda_report.html") -> str`: Exports an interactive HTML EDA report.
- `report() -> str`: Generates and prints a clean terminal summary.

---

## 3. AutoPrep

Leakage-free preprocessing, train/test splitting, and pipeline assembly.

```python
prep = AutoPrep(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
)
```

### Methods

- `inspect() -> Dict[str, Any]`: Identifies feature column types (numerical, categorical, datetime) and target characteristics.
- `preview() -> Dict[str, Any]`: Returns pipeline structure and split shapes before execution.
- `split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`: Creates reproducible `(X_train, X_test, y_train, y_test)`.
- `prepare() -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]`: Fits transformers strictly on training data and transforms both splits.
- `get_pipeline() -> ColumnTransformer`: Returns the fitted scikit-learn `ColumnTransformer`.
- `transform(X: pd.DataFrame) -> np.ndarray`: Transforms new/unseen feature DataFrames using the fitted pipeline.

---

## 4. AutoML

Automated model selection, cross-validation, hyperparameter tuning, and persistence.

```python
automl = AutoML(
    df: pd.DataFrame,
    target: str,
    task: Optional[str] = None,
    cv: int = 5,
    random_state: int = 42,
    n_iter: int = 10,
    scoring: Optional[str] = None
)
```

### Methods

- `detect_task() -> Dict[str, str]`: Automatically infers `'classification'` or `'regression'`.
- `inspect() -> Dict[str, Any]`: Returns detailed target distribution, feature summaries, and missingness metrics.
- `preview() -> List[str]`: Lists baseline candidate models for the detected task.
- `fit() -> Dict[str, Any]`: Fits all baseline models on training data and stores trained estimators.
- `evaluate() -> pd.DataFrame`: Performs cross-validation on all candidate models with fold-level metrics.
- `compare() -> pd.DataFrame`: Compares evaluated baseline models ordered by primary evaluation score.
- `tune(n_iter: Optional[int] = None) -> Dict[str, Any]`: Runs `RandomizedSearchCV` on candidate models without data leakage.
- `tuning_results() -> pd.DataFrame`: Tabulates best hyperparameters, CV scores, and tuning metrics.
- `best_models() -> Dict[str, Any]`: Returns the top performing model and configuration.
- `evaluate_tuned() -> pd.DataFrame`: Evaluates tuned models on held-out test data.
- `predict(X: Union[pd.DataFrame, pd.Series]) -> np.ndarray`: Predicts target values for raw input data.
- `predict_proba(X: Union[pd.DataFrame, pd.Series]) -> np.ndarray`: Predicts class probabilities for classification tasks.
- `save_model(filepath: str) -> None`: Serializes preprocessing pipeline, best model, and metadata to disk using joblib.
- `load_model(filepath: str) -> 'AutoML'`: Loads a saved serialized model pipeline for immediate inference.
- `model_info() -> Dict[str, Any]`: Returns rich metadata (task, target, model architecture, metrics, version).
