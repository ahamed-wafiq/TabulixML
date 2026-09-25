"""
cleaner.py -- Core AutoClean class for TabulixML.

Only depends on: pandas, numpy, scipy.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NumericalStrategy = Literal["mean", "median", "auto"]
CategoricalStrategy = Literal["mode"]
Strategy = Literal["mean", "median", "mode", "auto"]


class AutoClean:
    """
    Automatic cleaner for tabular pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to clean.
    numerical_strategy : {'mean', 'median', 'auto'}, default 'median'
        Imputation strategy for numerical missing values.
        'auto' uses median when skewness > 0.5, otherwise mean.
    categorical_strategy : {'mode'}, default 'mode'
        Imputation strategy for categorical missing values.
    remove_duplicates : bool, default True
        Whether to detect and drop duplicate rows.
    num_strategy : {'mean', 'median', 'auto'}, optional
        Alias for numerical_strategy for backwards compatibility.
    clean_col_names : bool, default False
        If True, strip, lower-case and replace spaces/special chars in
        column names.
    iqr_threshold : float, default 1.5
        Multiplier used for the IQR outlier fence.
        Outliers are *flagged*, never removed automatically.
    target : str, optional
        Optional target column name to analyze for class imbalance
        and possible target leakage.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_strategy: str = "median",
        categorical_strategy: str = "mode",
        remove_duplicates: bool = True,
        num_strategy: str | None = None,
        clean_col_names: bool = False,
        iqr_threshold: float = 1.5,
        target: str | None = None,
    ) -> None:
        # Validate df input
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Handle alias
        if num_strategy is not None:
            numerical_strategy = num_strategy

        # Validate numerical_strategy
        valid_num_strategies = ("mean", "median", "auto")
        if numerical_strategy not in valid_num_strategies:
            raise ValueError(
                f"Invalid numerical_strategy: '{numerical_strategy}'. "
                f"Must be one of {valid_num_strategies}."
            )

        # Validate categorical_strategy
        valid_cat_strategies = ("mode",)
        if categorical_strategy not in valid_cat_strategies:
            raise ValueError(
                f"Invalid categorical_strategy: '{categorical_strategy}'. "
                f"Must be one of {valid_cat_strategies}."
            )

        # Validate remove_duplicates
        if not isinstance(remove_duplicates, bool):
            raise TypeError("remove_duplicates must be a boolean (True or False).")

        # Validate target column if provided
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        self._original: pd.DataFrame = df.copy()
        self._df: pd.DataFrame = df.copy()
        self._numerical_strategy: str = numerical_strategy
        self._num_strategy: str = numerical_strategy  # backwards-compatible alias
        self._categorical_strategy: str = categorical_strategy
        self._remove_duplicates: bool = remove_duplicates
        self._clean_col_names: bool = clean_col_names
        self._iqr_threshold: float = iqr_threshold
        self._target: str | None = target

        self._history: list[dict] = []
        self._inspected: bool = False
        self._col_types: dict[str, str] = {}   # col -> "numerical" | "categorical" | "datetime"
        self._outlier_info: dict[str, dict] = {}
        # Populated after clean() so report() can show before/after
        self._last_clean_summary: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self, target: str | None = None) -> dict:
        """
        Analyse the DataFrame and return a summary dict.

        Parameters
        ----------
        target : str, optional
            Optional target column name to analyze for class imbalance
            and possible target leakage.

        Returns
        -------
        dict with keys:
            shape, dtypes, col_types, missing, duplicates,
            empty_cols, constant_cols, outliers,
            id_like_cols, high_cardinality_cols, high_cardinality,
            inconsistent_categories, inconsistent_cats,
            redundant_numerical, class_imbalance, target_leakage
        """
        if target is not None:
            if target not in self._df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame.")
            self._target = target
        effective_target = target or self._target

        self._col_types = self._detect_col_types(self._df)
        self._outlier_info = self._detect_outliers(self._df, self._col_types, self._iqr_threshold)
        self._inspected = True

        # Use positional access so duplicate column names don't produce
        # a DataFrame instead of a Series when indexing with df[col].
        missing: dict[str, int] = {}
        empty_cols: list[str] = []
        constant_cols: list[str] = []

        for i, col in enumerate(self._df.columns):
            s = self._df.iloc[:, i]
            null_count = int(s.isnull().sum())
            if null_count > 0:
                missing[col] = missing.get(col, 0) + null_count
            if s.isnull().all():
                if col not in empty_cols:
                    empty_cols.append(col)
            elif s.nunique(dropna=True) <= 1:
                if col not in constant_cols:
                    constant_cols.append(col)

        dup_count = int(self._df.duplicated().sum())

        id_like_cols = self._detect_id_like_cols(self._df, self._col_types)
        high_card_dict = self._detect_high_cardinality(self._df, self._col_types)
        high_card_cols = list(high_card_dict.keys())

        inconsistent_categories = self._detect_inconsistent_categories_detailed(self._df, self._col_types)
        inconsistent_cats = {
            col: sorted(list({v for entry in entries for v in entry["variants"]}))
            for col, entries in inconsistent_categories.items()
        }

        redundant_numerical = self._detect_redundant_numerical(self._df, self._col_types)

        class_imbalance = None
        target_leakage: list[dict] = []
        if effective_target is not None:
            class_imbalance, target_leakage = self._analyze_target(
                self._df, self._col_types, effective_target
            )

        summary = {
            "shape": self._df.shape,
            "dtypes": self._df.dtypes.astype(str).to_dict(),
            "col_types": dict(self._col_types),
            "missing": missing,
            "duplicates": dup_count,
            "empty_cols": empty_cols,
            "constant_cols": constant_cols,
            "outliers": {
                col: info for col, info in self._outlier_info.items()
                if info["count"] > 0
            },
            "id_like_cols": id_like_cols,
            "high_cardinality_cols": high_card_cols,
            "high_cardinality": high_card_dict,
            "inconsistent_categories": inconsistent_categories,
            "inconsistent_cats": inconsistent_cats,
            "redundant_numerical": redundant_numerical,
            "class_imbalance": class_imbalance,
            "target_leakage": target_leakage,
        }

        self._log(
            step="inspect",
            message="DataFrame inspected.",
            details=summary,
            operation="inspect",
            column=None,
            affected_count=0,
            strategy=None,
        )
        return summary

    def clean(self, approve: bool = True) -> pd.DataFrame | None:
        """
        Run all cleaning steps and return the cleaned DataFrame.

        Parameters
        ----------
        approve : bool, default True
            If False, run preview() to show proposed changes and return None
            without modifying anything.  If True (default), apply all changes.

        Steps (in order):
          1. Optionally clean column names.
          2. Drop completely empty columns.
          3. Drop duplicate rows (if remove_duplicates=True).
          4. Impute missing values according to configured strategies.

        Returns
        -------
        pd.DataFrame -- a cleaned copy (original is never mutated), or None
        when approve=False.
        """
        if not approve:
            self.preview()
            return None

        if not self._inspected:
            self.inspect()

        rows_before = len(self._df)
        cols_before = len(self._df.columns)
        df = self._df.copy()

        # 1. Clean column names
        col_names_changed: list[tuple[str, str]] = []
        if self._clean_col_names:
            old_names = list(df.columns)
            df = self._apply_clean_col_names(df)
            col_names_changed = [
                (old, new)
                for old, new in zip(old_names, df.columns)
                if old != new
            ]
            if col_names_changed:
                self._log(
                    step="clean_col_names",
                    message=f"Renamed {len(col_names_changed)} column(s).",
                    details={"renames": col_names_changed, "col": [o for o, _ in col_names_changed]},
                    operation="clean_col_names",
                    column=[o for o, _ in col_names_changed],
                    affected_count=len(col_names_changed),
                    strategy=None,
                )

        # 2. Drop empty columns
        empty_cols = [c for c in df.columns if df[c].isnull().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self._log(
                step="drop_empty_cols",
                message=f"Dropped {len(empty_cols)} empty column(s).",
                details={"cols": empty_cols, "col": empty_cols},
                operation="drop_empty_cols",
                column=empty_cols,
                affected_count=len(empty_cols),
                strategy=None,
            )

        # 3. Drop duplicate rows (if enabled)
        removed_dups = 0
        if self._remove_duplicates:
            before_dup = len(df)
            df = df.drop_duplicates()
            removed_dups = before_dup - len(df)
            if removed_dups:
                self._log(
                    step="drop_duplicates",
                    message=f"Removed {removed_dups} duplicate row(s).",
                    details={"removed": removed_dups, "col": None},
                    operation="drop_duplicates",
                    column=None,
                    affected_count=removed_dups,
                    strategy=None,
                )

        # 4. Impute missing values
        imputed_before = len(self._history)
        df = self._impute(df)
        imputed_steps = [
            e for e in self._history[imputed_before:]
            if e["step"] in ("impute_numerical", "impute_categorical")
        ]

        rows_after = len(df)
        cols_after = len(df.columns)

        # Store summary for report()
        self._last_clean_summary = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "empty_cols_dropped": empty_cols,
            "duplicate_rows_removed": removed_dups,
            "col_names_changed": col_names_changed,
            "imputed": [
                {
                    "col": e["details"]["col"],
                    "type": "numerical" if e["step"] == "impute_numerical" else "categorical",
                    "fill_value": e["details"]["fill_value"],
                    "missing_count": e["details"]["missing_count"],
                    "strategy": e.get("strategy"),
                }
                for e in imputed_steps
            ],
        }

        # Update internal state so report() reflects cleaned data
        self._df = df
        self._inspected = False   # Force re-inspect on next explicit call
        return df.copy()

    def reset(self) -> AutoClean:
        """
        Reset cleaner state back to the original DataFrame.

        Clears cleaning history and allows the cleaner to inspect, preview,
        and clean the original data again.

        Returns
        -------
        self : AutoClean
        """
        self._df = self._original.copy()
        self._history.clear()
        self._inspected = False
        self._col_types.clear()
        self._outlier_info.clear()
        self._last_clean_summary.clear()
        return self

    def report(self, target: str | None = None) -> None:
        """
        Print a human-readable before/after cleaning summary to stdout.

        If clean() has been called, the report shows:
          - BEFORE shape  (original DataFrame as supplied)
          - AFTER  shape  (after all cleaning steps)
          - Every structural change: columns dropped, duplicates removed,
            column renames, and per-column imputation details.
          - Quality checks & warnings: ID-like columns, high-cardinality columns,
            inconsistent categorical values, redundant numerical features,
            outliers, and (if target provided) class distribution/leakage.

        If clean() has NOT been called, only the current inspection
        summary is shown.
        """
        if target is not None:
            if target not in self._original.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame.")
            self._target = target
        effective_target = target or self._target

        orig_info = AutoClean._quick_inspect(self._original)
        has_been_cleaned = bool(self._last_clean_summary)

        sep = "=" * 60
        thin = "-" * 60
        print(sep)
        print("  TabulixML -- AutoClean Report")
        print(sep)

        # -- Before / After shape ---------------------------------------
        if has_been_cleaned:
            cs = self._last_clean_summary
            print(f"  {'':12}  {'Rows':>6}   {'Cols':>5}")
            print(f"  {'BEFORE':12}  {cs['rows_before']:>6}   {cs['cols_before']:>5}")
            print(f"  {'AFTER':12}  {cs['rows_after']:>6}   {cs['cols_after']:>5}")
            row_delta = cs['rows_after'] - cs['rows_before']
            col_delta = cs['cols_after'] - cs['cols_before']
            row_str = f"+{row_delta}" if row_delta >= 0 else str(row_delta)
            col_str = f"+{col_delta}" if col_delta >= 0 else str(col_delta)
            print(f"  {'DELTA':12}  {row_str:>7}   {col_str:>5}")
        else:
            rows, cols = orig_info["shape"]
            print(f"  Shape (before) : {rows} rows x {cols} cols")
            print("  (call clean() to see before/after comparison)")

        # -- Column-type breakdown (original) ---------------------------
        type_counts: dict[str, int] = {}
        for t in orig_info["col_types"].values():
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\n  Column types   : {type_counts}")

        # -- Missing values (original) -----------------------------------
        if orig_info["missing"]:
            print("\n  Missing Values (before cleaning):")
            total_rows = orig_info["shape"][0]
            for col, cnt in orig_info["missing"].items():
                pct = 100 * cnt / total_rows
                print(f"    {col:<30}  {cnt:>6} ({pct:.1f}%)")
        else:
            print("  Missing Values : none")

        # -- Empty / constant columns (original) ------------------------
        if orig_info["empty_cols"]:
            print(f"\n  Empty columns  : {orig_info['empty_cols']}")
        if orig_info["constant_cols"]:
            print(f"  Constant cols  : {orig_info['constant_cols']}")

        # -- Duplicates (original) ---------------------------------------
        print(f"  Duplicates     : {orig_info['duplicates']}")

        # -- Changes made by clean() ------------------------------------
        if has_been_cleaned:
            print(f"\n{thin}")
            print("  Changes Made by clean():")
            cs = self._last_clean_summary
            any_change = False

            if cs["col_names_changed"]:
                any_change = True
                print(f"  * Renamed {len(cs['col_names_changed'])} column(s):")
                for old, new in cs["col_names_changed"]:
                    print(f"      '{old}'  ->  '{new}'")

            if cs["empty_cols_dropped"]:
                any_change = True
                print(f"  * Dropped {len(cs['empty_cols_dropped'])} empty column(s): "
                      f"{cs['empty_cols_dropped']}")

            if cs["duplicate_rows_removed"]:
                any_change = True
                print(f"  * Removed {cs['duplicate_rows_removed']} duplicate row(s).")
            elif not self._remove_duplicates and orig_info["duplicates"] > 0:
                print(f"  * Duplicate removal disabled: kept {orig_info['duplicates']} duplicate row(s).")

            if cs["imputed"]:
                any_change = True
                print(f"  * Imputed missing values in {len(cs['imputed'])} column(s):")
                for imp in cs["imputed"]:
                    fill = imp['fill_value']
                    fill_str = (
                        f"{fill:.4g}" if isinstance(fill, float) else f"'{fill}'"
                    )
                    strat_str = f" [{imp['strategy']}]" if imp.get('strategy') else ""
                    print(f"      {imp['col']:<28}  "
                          f"[{imp['type']}{strat_str}]  "
                          f"{imp['missing_count']} value(s) -> {fill_str}")

            if not any_change:
                print("  * No changes were necessary.")

        # -- Data Quality Warnings (flagged only, never removed) --------
        print(f"\n{thin}")
        orig_col_types = orig_info["col_types"]

        # ID-like columns
        id_like_cols = AutoClean._detect_id_like_cols(self._original, orig_col_types)
        if id_like_cols:
            print("  Possible ID column(s) (flagged only, NOT removed):")
            for c in id_like_cols:
                print(f"    * '{c}' (unique values close to row count)")

        # High cardinality
        high_card = AutoClean._detect_high_cardinality(self._original, orig_col_types)
        if high_card:
            print("  High-cardinality categorical column(s) (flagged only, NOT removed):")
            for c, cnt in high_card.items():
                print(f"    * '{c}': {cnt} unique categories")

        # Inconsistent categories
        inconsistent_detailed = AutoClean._detect_inconsistent_categories_detailed(
            self._original, orig_col_types
        )
        if inconsistent_detailed:
            print("  Inconsistent categorical values (flagged only, NOT changed):")
            for col, clusters in inconsistent_detailed.items():
                for cluster in clusters:
                    sample = cluster["variants"][:5]
                    more = f" (+{len(cluster['variants'])-5} more)" if len(cluster['variants']) > 5 else ""
                    print(f"    * '{col}' variants: {sample}{more} -> suggested: '{cluster['suggested']}'")

        # Redundant numerical features
        redundant = AutoClean._detect_redundant_numerical(self._original, orig_col_types)
        if redundant:
            print("  Redundant numerical feature(s) (flagged only, NOT removed):")
            for pair in redundant:
                print(f"    * '{pair['col1']}' <-> '{pair['col2']}' (correlation = {pair['correlation']:.4f})")

        # Target analysis
        if effective_target:
            cb, leak = AutoClean._analyze_target(
                self._original, orig_col_types, effective_target
            )
            print(f"  Target Analysis ('{effective_target}'):")
            print("    Class distribution:")
            for cls_val, cnt in cb["counts"].items():
                pct = cb["percentages"].get(cls_val, 0)
                print(f"      '{cls_val}': {cnt} ({pct:.1f}%)")
            if cb.get("is_imbalanced"):
                print(f"    * Warning: Class imbalance detected in target '{effective_target}'.")
            if leak:
                print("    * Warning: Possible target leakage detected:")
                for l in leak:
                    print(f"        '{l['col']}': {l['details']}")

        # Outliers (flagged only, always from original)
        orig_outliers = AutoClean._detect_outliers(
            self._original, orig_col_types, self._iqr_threshold
        )
        orig_outliers_with_count = {
            col: info for col, info in orig_outliers.items() if info["count"] > 0
        }
        if orig_outliers_with_count:
            print("  Outliers (IQR - flagged only, NOT removed):")
            for col, o in orig_outliers_with_count.items():
                print(f"    {col:<30}  {o['count']:>4} outlier(s)  "
                      f"fence [{o['lower_fence']:.3g}, {o['upper_fence']:.3g}]")
        else:
            print("  Outliers       : none detected")

        print(sep)

    def history(self) -> list[dict]:
        """
        Return the list of cleaning steps performed so far.

        Each entry is a dict with keys:
            operation        : name of the cleaning operation
            column           : affected column name(s) or None
            affected_count   : count of affected values or rows
            strategy         : strategy used ('mean', 'median', 'mode') or None
            step             : step identifier
            timestamp        : ISO 8601 formatted timestamp
            message          : descriptive message
            details          : full details dictionary
        """
        return list(self._history)

    def preview(self, target: str | None = None) -> dict:
        """
        Analyse the DataFrame and print a human-readable plan of ALL changes
        that clean() would apply -- without modifying anything.
        """
        if target is not None:
            if target not in self._original.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame.")
            self._target = target
        effective_target = target or self._target

        df = self._original.copy()   # work on a safe copy -- never mutates
        col_types = self._detect_col_types(df)
        qi = self._quick_inspect(df)

        # -- 1. Column renames (if clean_col_names is on) ------------------
        col_names_to_rename: list[tuple[str, str]] = []
        if self._clean_col_names:
            renamed_df = self._apply_clean_col_names(df)
            col_names_to_rename = [
                (old, new)
                for old, new in zip(df.columns, renamed_df.columns)
                if old != new
            ]

        # -- 2. Imputation plan (after simulating empty-col drop + dedup) --
        sim = df.copy()
        # simulate empty-col drop
        sim = sim.drop(columns=qi["empty_cols"], errors="ignore")
        # simulate dedup only if remove_duplicates is enabled
        if self._remove_duplicates:
            sim = sim.drop_duplicates()
        sim_types = self._detect_col_types(sim)

        impute_plan: list[dict] = []
        for col in sim.columns:
            missing_count = int(sim[col].isnull().sum())
            if missing_count == 0:
                continue
            ctype = sim_types.get(col, "categorical")
            if ctype == "numerical":
                fill_val = self._numerical_fill(sim[col])
                strategy = self._numerical_strategy
                if strategy == "auto":
                    clean_s = sim[col].dropna()
                    if len(clean_s) >= 3:
                        from scipy import stats as _stats
                        skewness = float(abs(_stats.skew(clean_s)))
                        strategy = "median" if (not np.isnan(skewness) and skewness > 0.5) else "mean"
                    else:
                        strategy = "mean"
                impute_plan.append({
                    "col": col,
                    "type": "numerical",
                    "strategy": strategy,
                    "fill_value": fill_val,
                    "missing_count": missing_count,
                })
            elif ctype == "categorical":
                mode_vals = sim[col].mode(dropna=True)
                fill_val_cat = mode_vals.iloc[0] if not mode_vals.empty else None
                impute_plan.append({
                    "col": col,
                    "type": "categorical",
                    "strategy": self._categorical_strategy,
                    "fill_value": fill_val_cat,
                    "missing_count": missing_count,
                })

        # -- 3. Inconsistent categories (casing/whitespace variants) -------
        inconsistent_categories = self._detect_inconsistent_categories_detailed(df, col_types)
        inconsistent_cats = {
            col: sorted(list({v for entry in entries for v in entry["variants"]}))
            for col, entries in inconsistent_categories.items()
        }

        # -- 4. Outliers (from original; never removed) --------------------
        outliers = {
            col: info
            for col, info in self._detect_outliers(df, col_types, self._iqr_threshold).items()
            if info["count"] > 0
        }

        # -- 5. ID-like, high-cardinality, redundant, target checks --------
        id_like_cols = self._detect_id_like_cols(df, col_types)
        high_card_dict = self._detect_high_cardinality(df, col_types)
        high_card_cols = list(high_card_dict.keys())
        redundant_numerical = self._detect_redundant_numerical(df, col_types)

        class_imbalance = None
        target_leakage: list[dict] = []
        if effective_target:
            class_imbalance, target_leakage = self._analyze_target(
                df, col_types, effective_target
            )

        dup_rows_to_drop = qi["duplicates"] if self._remove_duplicates else 0

        plan = {
            "col_names_to_rename":       col_names_to_rename,
            "empty_cols_to_drop":        qi["empty_cols"],
            "constant_cols":             qi["constant_cols"],
            "duplicate_rows":            dup_rows_to_drop,
            "raw_duplicate_rows":        qi["duplicates"],
            "remove_duplicates_enabled": self._remove_duplicates,
            "impute_plan":               impute_plan,
            "inconsistent_cats":         inconsistent_cats,
            "inconsistent_categories":   inconsistent_categories,
            "outliers":                  outliers,
            "id_like_cols":              id_like_cols,
            "high_cardinality_cols":     high_card_cols,
            "high_cardinality":          high_card_dict,
            "redundant_numerical":       redundant_numerical,
            "class_imbalance":           class_imbalance,
            "target_leakage":            target_leakage,
        }

        self._print_preview(plan, qi)
        return plan

    # ------------------------------------------------------------------
    # Private: preview printer
    # ------------------------------------------------------------------

    @staticmethod
    def _print_preview(plan: dict, qi: dict) -> None:
        """Print the preview plan in a readable table format."""
        sep  = "=" * 60
        thin = "-" * 60
        rows, cols = qi["shape"]

        print(sep)
        print("  TabulixML -- AutoClean Preview")
        print("  (no changes applied yet)")
        print(sep)
        print(f"  DataFrame : {rows} rows x {cols} columns")
        print(thin)

        any_change = False

        # Column renames
        if plan["col_names_to_rename"]:
            any_change = True
            print(f"  [RENAME] {len(plan['col_names_to_rename'])} column name(s) will be cleaned:")
            for old, new in plan["col_names_to_rename"]:
                print(f"      '{old}'  ->  '{new}'")

        # Empty columns
        if plan["empty_cols_to_drop"]:
            any_change = True
            print(f"  [DROP]   {len(plan['empty_cols_to_drop'])} completely empty column(s):")
            for c in plan["empty_cols_to_drop"]:
                print(f"      '{c}'")

        # Constant columns
        if plan["constant_cols"]:
            print(f"  [FLAG]   {len(plan['constant_cols'])} constant column(s) (flagged, not removed):")
            for c in plan["constant_cols"]:
                print(f"      '{c}'")

        # ID-like columns
        if plan.get("id_like_cols"):
            print(f"  [FLAG]   {len(plan['id_like_cols'])} possible ID column(s) (unique values close to row count, not removed):")
            for c in plan["id_like_cols"]:
                print(f"      '{c}'")

        # Duplicates
        if plan.get("remove_duplicates_enabled", True):
            if plan["duplicate_rows"] > 0:
                any_change = True
                print(f"  [DROP]   {plan['duplicate_rows']} duplicate row(s) will be removed.")
        elif plan.get("raw_duplicate_rows", 0) > 0:
            print(f"  [INFO]   {plan['raw_duplicate_rows']} duplicate row(s) detected (duplicate removal disabled).")

        # Imputation plan
        if plan["impute_plan"]:
            any_change = True
            print(f"  [IMPUTE] {len(plan['impute_plan'])} column(s) have missing values:")
            for p in plan["impute_plan"]:
                fill = p["fill_value"]
                fill_str = (
                    f"{fill:.4g}" if isinstance(fill, float) else
                    f"'{fill}'"    if fill is not None else "(no mode)"
                )
                print(f"      {p['col']:<28} [{p['type']:>11}]  "
                      f"{p['missing_count']} missing  strategy={p['strategy']}  fill={fill_str}")

        # Inconsistent categories (with suggested normalized value)
        inconsistent_detailed = plan.get("inconsistent_categories") or {}
        if inconsistent_detailed:
            print(f"  [WARN]   {len(inconsistent_detailed)} column(s) have inconsistent categorical values:")
            for col, clusters in inconsistent_detailed.items():
                for cluster in clusters:
                    sample = cluster["variants"][:5]
                    more   = f" (+{len(cluster['variants'])-5} more)" if len(cluster['variants']) > 5 else ""
                    print(f"      '{col}' variants: {sample}{more}  ->  suggested: '{cluster['suggested']}'")
        elif plan.get("inconsistent_cats"):
            print(f"  [WARN]   {len(plan['inconsistent_cats'])} column(s) have inconsistent casing:")
            for col, variants in plan["inconsistent_cats"].items():
                sample = variants[:5]
                more   = f" (+{len(variants)-5} more)" if len(variants) > 5 else ""
                print(f"      '{col}' variants: {sample}{more}")

        # High-cardinality categorical columns
        if plan.get("high_cardinality_cols"):
            print(f"  [WARN]   {len(plan['high_cardinality_cols'])} high-cardinality categorical column(s):")
            high_card = plan.get("high_cardinality", {})
            for c in plan["high_cardinality_cols"]:
                cnt = high_card.get(c, "many")
                print(f"      '{c}' has {cnt} unique categories")

        # Redundant numerical features
        if plan.get("redundant_numerical"):
            print(f"  [WARN]   {len(plan['redundant_numerical'])} redundant numerical feature pair(s) (|r| >= 0.90):")
            for pair in plan["redundant_numerical"]:
                print(f"      '{pair['col1']}' <-> '{pair['col2']}' (correlation = {pair['correlation']:.4f})")

        # Class imbalance and target leakage (if target provided)
        if plan.get("class_imbalance"):
            cb = plan["class_imbalance"]
            print(f"  [INFO]   Target column: '{cb['target']}' class distribution:")
            for cls_val, cnt in cb["counts"].items():
                pct = cb["percentages"].get(cls_val, 0)
                print(f"      '{cls_val}': {cnt} ({pct:.1f}%)")
            if cb.get("is_imbalanced"):
                print(f"  [WARN]   Class imbalance detected in target '{cb['target']}'.")

        if plan.get("target_leakage"):
            print(f"  [WARN]   {len(plan['target_leakage'])} possible target leakage column(s):")
            for leak in plan["target_leakage"]:
                print(f"      '{leak['col']}': {leak['details']}")

        # Outliers
        if plan["outliers"]:
            print("  [WARN]   Outliers detected (IQR - NOT removed automatically):")
            for col, o in plan["outliers"].items():
                print(f"      {col:<28}  {o['count']:>3} outlier(s)  "
                      f"fence [{o['lower_fence']:.3g}, {o['upper_fence']:.3g}]")

        has_warnings = (
            bool(plan.get("inconsistent_cats")) or
            bool(plan.get("outliers")) or
            bool(plan.get("id_like_cols")) or
            bool(plan.get("high_cardinality_cols")) or
            bool(plan.get("redundant_numerical")) or
            bool(plan.get("target_leakage"))
        )

        if not any_change and not has_warnings:
            print("  No issues detected. DataFrame appears clean.")

        print(sep)

    # ------------------------------------------------------------------
    # Private class-level helpers (usable without a live instance)
    # ------------------------------------------------------------------

    @classmethod
    def _quick_inspect(cls, df: pd.DataFrame) -> dict:
        """Lightweight inspect that does not touch self._history."""
        col_types = cls._detect_col_types(df)
        missing: dict[str, int] = {}
        empty_cols: list[str] = []
        constant_cols: list[str] = []
        for i, col in enumerate(df.columns):
            s = df.iloc[:, i]
            nc = int(s.isnull().sum())
            if nc > 0:
                missing[col] = missing.get(col, 0) + nc
            if s.isnull().all():
                if col not in empty_cols:
                    empty_cols.append(col)
            elif s.nunique(dropna=True) <= 1:
                if col not in constant_cols:
                    constant_cols.append(col)
        return {
            "shape": df.shape,
            "col_types": col_types,
            "missing": missing,
            "duplicates": int(df.duplicated().sum()),
            "empty_cols": empty_cols,
            "constant_cols": constant_cols,
        }

    # ------------------------------------------------------------------
    # Private detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_id_like_cols(df: pd.DataFrame, col_types: dict[str, str]) -> list[str]:
        """
        Detect columns that behave like identifiers.
        Criteria:
          - High unique value ratio close to row count (>= 0.90)
          - Row count >= 3 and at least 3 unique non-null values
          - Excludes datetimes and non-integer continuous floats unless named like an ID
          - If not ID-named, requires at least 10 rows
        """
        id_like: list[str] = []
        n_rows = len(df)
        if n_rows < 3:
            return id_like

        for i, col in enumerate(df.columns):
            s = df.iloc[:, i]
            if s.isnull().all():
                continue
            nunique = int(s.nunique(dropna=True))
            if nunique < 3:
                continue

            ratio = nunique / n_rows
            if ratio < 0.90:
                continue

            ctype = col_types.get(col, "")
            if ctype == "datetime":
                continue

            col_lower = str(col).lower()
            is_id_named = any(token in col_lower for token in ("id", "key", "uuid", "code", "token"))

            if not is_id_named and n_rows < 10:
                continue

            # Continuous floats without integer values or ID names are usually measurements
            if pd.api.types.is_float_dtype(s) and not is_id_named:
                non_null = s.dropna()
                if not (non_null % 1 == 0).all():
                    continue

            if col not in id_like:
                id_like.append(col)
        return id_like

    @staticmethod
    def _detect_high_cardinality(
        df: pd.DataFrame,
        col_types: dict[str, str],
        threshold: int = 10,
    ) -> dict[str, int]:
        """
        Flag categorical columns with unusually many unique values.
        Returns {col: unique_count} for columns where nunique >= threshold.
        """
        result: dict[str, int] = {}
        for i, col in enumerate(df.columns):
            if col_types.get(col) != "categorical":
                continue
            s = df.iloc[:, i]
            if s.isnull().all():
                continue
            nunique = int(s.nunique(dropna=True))
            if nunique <= 1:
                continue
            if nunique >= threshold:
                result[col] = nunique
        return result

    @classmethod
    def _detect_inconsistent_categories_detailed(
        cls,
        df: pd.DataFrame,
        col_types: dict[str, str],
    ) -> dict[str, list[dict]]:
        """
        For each categorical column, find groups of values that differ in
        casing or whitespace variants, along with suggested normalized values.
        Returns {col: [{"variants": [...], "suggested": str, "counts": {...}}]}
        """
        result: dict[str, list[dict]] = {}
        for i, col in enumerate(df.columns):
            if col_types.get(col) != "categorical":
                continue
            series = df.iloc[:, i].dropna()
            if series.empty:
                continue

            counts: dict[str, int] = {}
            for v in series:
                raw = str(v)
                counts[raw] = counts.get(raw, 0) + 1

            groups: dict[str, list[str]] = {}
            for raw in counts:
                clean_str = " ".join(raw.strip().split())
                key = clean_str.lower()
                groups.setdefault(key, []).append(raw)

            col_clusters: list[dict] = []
            for key, raw_list in groups.items():
                has_mult = len(raw_list) > 1
                has_ws = any(r != " ".join(r.strip().split()) for r in raw_list)
                if has_mult or has_ws:
                    best_raw = max(raw_list, key=lambda r: counts[r])
                    cleaned_best = " ".join(best_raw.strip().split())

                    # Check for Title Case or uppercase acronym
                    title_candidates = [
                        " ".join(r.strip().split())
                        for r in raw_list
                        if " ".join(r.strip().split()).istitle()
                    ]
                    if title_candidates:
                        suggested = max(title_candidates, key=lambda t: counts.get(t, 0))
                    elif cleaned_best.islower() and key.title() != key:
                        suggested = key.title()
                    else:
                        suggested = cleaned_best

                    col_clusters.append({
                        "variants": sorted(raw_list),
                        "suggested": suggested,
                        "counts": {r: counts[r] for r in raw_list},
                    })

            if col_clusters:
                result[col] = col_clusters
        return result

    @classmethod
    def _detect_inconsistent_categories(
        cls,
        df: pd.DataFrame,
        col_types: dict[str, str],
    ) -> dict[str, list[str]]:
        """Flat mapping of col -> list of raw variants (for backward compatibility)."""
        detailed = cls._detect_inconsistent_categories_detailed(df, col_types)
        return {
            col: sorted(list({v for entry in entries for v in entry["variants"]}))
            for col, entries in detailed.items()
        }

    @staticmethod
    def _detect_redundant_numerical(
        df: pd.DataFrame,
        col_types: dict[str, str],
        threshold: float = 0.90,
    ) -> list[dict]:
        """
        Detect pairs of numerical columns with high Pearson correlation (|r| >= threshold).
        """
        redundant: list[dict] = []
        num_indices = [i for i, c in enumerate(df.columns) if col_types.get(c) == "numerical"]
        if len(num_indices) < 2:
            return redundant

        for idx_a in range(len(num_indices)):
            for idx_b in range(idx_a + 1, len(num_indices)):
                i, j = num_indices[idx_a], num_indices[idx_b]
                c1, c2 = df.columns[i], df.columns[j]
                if c1 == c2:
                    continue
                s1, s2 = df.iloc[:, i], df.iloc[:, j]
                pair_df = pd.DataFrame({"s1": s1, "s2": s2}).dropna()
                if len(pair_df) < 3:
                    continue
                if pair_df["s1"].std() == 0 or pair_df["s2"].std() == 0:
                    continue
                r = float(pair_df["s1"].corr(pair_df["s2"]))
                if abs(r) >= threshold:
                    redundant.append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": round(r, 4),
                    })
        return redundant

    @staticmethod
    def _analyze_target(
        df: pd.DataFrame,
        col_types: dict[str, str],
        target: str,
    ) -> tuple[dict, list[dict]]:
        """
        Analyze class distribution/imbalance and possible target leakage for a given target column.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        target_idx = list(df.columns).index(target)
        s_target = df.iloc[:, target_idx]
        total = len(df)
        counts_s = s_target.value_counts(dropna=False)
        counts = {str(k): int(v) for k, v in counts_s.items()}
        percentages = {
            str(k): round(float(v) * 100.0 / total, 2)
            for k, v in counts_s.items()
        }
        is_imbalanced = False
        if len(percentages) > 1:
            max_pct = max(percentages.values())
            if max_pct >= 70.0:
                is_imbalanced = True

        class_imbalance = {
            "target": target,
            "counts": counts,
            "percentages": percentages,
            "is_imbalanced": is_imbalanced,
        }

        # Target leakage detection
        leakage: list[dict] = []
        for i, col in enumerate(df.columns):
            if i == target_idx or col == target:
                continue
            s_col = df.iloc[:, i]
            if s_col.isnull().all():
                continue

            pair_df = pd.DataFrame({"col": s_col, "target": s_target}).dropna()
            if len(pair_df) == 0:
                continue

            # 1. Exact duplicate
            if (pair_df["col"] == pair_df["target"]).all():
                leakage.append({
                    "col": col,
                    "reason": "duplicate",
                    "details": f"Column '{col}' directly duplicates target '{target}'."
                })
                continue

            # 2. String normalized duplicate
            if (pair_df["col"].astype(str).str.strip().str.lower() == pair_df["target"].astype(str).str.strip().str.lower()).all():
                leakage.append({
                    "col": col,
                    "reason": "duplicate",
                    "details": f"Column '{col}' duplicates target '{target}' after casing/whitespace normalization."
                })
                continue

            # 3. High numerical correlation (|r| >= 0.95)
            if pd.api.types.is_numeric_dtype(s_col) and pd.api.types.is_numeric_dtype(s_target):
                if len(pair_df) >= 3 and pair_df["col"].std() > 0 and pair_df["target"].std() > 0:
                    r = float(pair_df["col"].corr(pair_df["target"]))
                    if abs(r) >= 0.95:
                        leakage.append({
                            "col": col,
                            "reason": "high_correlation",
                            "details": f"High numerical correlation with target '{target}' (r = {r:.4f})."
                        })
                        continue

            # 4. Perfect 1:1 categorical association (for discrete features with enough rows)
            if len(pair_df) >= 5 and pair_df["col"].nunique() > 1:
                targets_per_val = pair_df.groupby("col")["target"].nunique()
                if (targets_per_val == 1).all() and pair_df["col"].nunique() <= pair_df["target"].nunique():
                    leakage.append({
                        "col": col,
                        "reason": "perfect_mapping",
                        "details": f"Column '{col}' has a 1:1 mapping with target '{target}' (possible proxy)."
                    })
                    continue

        return class_imbalance, leakage

    @staticmethod
    def _detect_col_types(df: pd.DataFrame) -> dict[str, str]:
        """Classify every column as 'numerical', 'datetime', or 'categorical'."""
        col_types: dict[str, str] = {}
        for i, col in enumerate(df.columns):
            s = df.iloc[:, i]
            if pd.api.types.is_datetime64_any_dtype(s):
                col_types[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(s):
                col_types[col] = "numerical"
            else:
                # Attempt datetime parse on object columns
                sample = s.dropna().head(50)
                if len(sample) == 0:
                    col_types[col] = "categorical"
                    continue
                try:
                    parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                    if parsed.notna().all():
                        col_types[col] = "datetime"
                    else:
                        col_types[col] = "categorical"
                except (ValueError, TypeError):
                    col_types[col] = "categorical"
        return col_types

    @staticmethod
    def _detect_outliers(
        df: pd.DataFrame,
        col_types: dict[str, str],
        threshold: float,
    ) -> dict[str, dict]:
        """IQR-based outlier detection for numerical columns."""
        result: dict[str, dict] = {}
        for i, col in enumerate(df.columns):
            ctype = col_types.get(col)
            if ctype != "numerical":
                continue
            series = df.iloc[:, i].dropna()
            if series.empty:
                continue
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask = (series < lower) | (series > upper)
            result[col] = {
                "count": int(outlier_mask.sum()),
                "lower_fence": lower,
                "upper_fence": upper,
                "q1": q1,
                "q3": q3,
            }
        return result

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values according to column type and chosen strategy."""
        col_types = self._detect_col_types(df)
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count == 0:
                continue

            ctype = col_types.get(col, "categorical")

            if ctype == "numerical":
                fill_val = self._numerical_fill(df[col])
                strategy = self._numerical_strategy
                if strategy == "auto":
                    clean_s = df[col].dropna()
                    if len(clean_s) >= 3:
                        skewness = float(abs(stats.skew(clean_s)))
                        strategy = "median" if (not np.isnan(skewness) and skewness > 0.5) else "mean"
                    else:
                        strategy = "mean"

                df[col] = df[col].fillna(fill_val)
                self._log(
                    step="impute_numerical",
                    message=f"Imputed {missing_count} missing value(s) in '{col}' with {fill_val:.4g}.",
                    details={
                        "col": col,
                        "column": col,
                        "fill_value": fill_val,
                        "missing_count": missing_count,
                        "affected": missing_count,
                        "affected_count": missing_count,
                        "strategy": strategy,
                    },
                    operation="impute_numerical",
                    column=col,
                    affected_count=missing_count,
                    strategy=strategy,
                )

            elif ctype == "categorical":
                mode_vals = df[col].mode(dropna=True)
                if mode_vals.empty:
                    continue
                fill_val_cat = mode_vals.iloc[0]
                df[col] = df[col].fillna(fill_val_cat)
                self._log(
                    step="impute_categorical",
                    message=f"Imputed {missing_count} missing value(s) in '{col}' with mode '{fill_val_cat}'.",
                    details={
                        "col": col,
                        "column": col,
                        "fill_value": str(fill_val_cat),
                        "missing_count": missing_count,
                        "affected": missing_count,
                        "affected_count": missing_count,
                        "strategy": self._categorical_strategy,
                    },
                    operation="impute_categorical",
                    column=col,
                    affected_count=missing_count,
                    strategy=self._categorical_strategy,
                )
            # datetime: leave as-is (no imputation without domain knowledge)

        return df

    def _numerical_fill(self, series: pd.Series) -> float:
        """Compute the fill value for a numerical series."""
        clean_s = series.dropna()
        if clean_s.empty:
            return 0.0
        strategy = self._numerical_strategy
        if strategy == "auto":
            if len(clean_s) >= 3:
                skewness = float(abs(stats.skew(clean_s)))
                strategy = "median" if (not np.isnan(skewness) and skewness > 0.5) else "mean"
            else:
                strategy = "mean"
        if strategy == "median":
            return float(clean_s.median())
        return float(clean_s.mean())   # "mean"

    @staticmethod
    def _apply_clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
        """Strip, lower-case and slug-ify column names."""
        def slugify(name: str) -> str:
            name = str(name).strip().lower()
            name = re.sub(r"[\s\-]+", "_", name)
            name = re.sub(r"[^\w]", "", name)
            name = re.sub(r"_+", "_", name)
            return name.strip("_") or "col"

        new_names = [slugify(c) for c in df.columns]
        # Deduplicate
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for n in new_names:
            if n in seen:
                seen[n] += 1
                deduped.append(f"{n}_{seen[n]}")
            else:
                seen[n] = 0
                deduped.append(n)
        df = df.copy()
        df.columns = deduped
        return df

    def _log(
        self,
        step: str,
        message: str,
        details: dict | None = None,
        operation: str | None = None,
        column: str | list[str] | None = None,
        affected_count: int | None = None,
        strategy: str | None = None,
    ) -> None:
        det = details or {}
        op = operation or det.get("operation", step)
        col = column if column is not None else det.get("column", det.get("col"))
        aff = affected_count if affected_count is not None else det.get(
            "affected_count", det.get("affected", det.get("missing_count", det.get("removed", 0)))
        )
        strat = strategy if strategy is not None else det.get("strategy")

        self._history.append({
            "step": step,
            "operation": op,
            "column": col,
            "affected_count": aff,
            "affected": aff,
            "strategy": strat,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": message,
            "details": det,
        })
