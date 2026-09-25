"""
src/datacraft/eda.py -- AutoEDA Module for DataCraft

Provides fast, read-only exploratory data analysis for tabular datasets:
  - inspect()     : Basic dimensions, column names, data types, missing values,
                    unique value counts, and duplicate row count.
  - summary()     : Descriptive statistical summaries for numerical and categorical features.
  - correlations(): Numerical correlation matrix and identification of highly correlated pairs.
  - quality()     : Consolidated data-quality audit (missing rates, duplicates, constant columns,
                    high-cardinality features, numerical outliers).
  - report()      : Clean, human-readable terminal report combining all findings.
  - visualize()   : Automatically generate histograms, boxplots, category bars, and correlation heatmap.
  - save_report() : Export a self-contained, standalone HTML report with tables, warnings, and embedded plots.

Guarantees:
  - AutoEDA is strictly read-only and NEVER modifies the caller's DataFrame.
  - Uses pandas and numpy for primary analysis and reuses existing DataCraft
    detection logic for consistency across AutoClean and AutoEDA.
"""

import base64
import html
import io
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datacraft.cleaner import AutoClean


def _sanitize_name(name: Any) -> str:
    """Sanitize column name for safe filenames."""
    s = re.sub(r"[^\w\-]", "_", str(name)).strip("_")
    return s if s else "col"


class AutoEDA:
    """
    Automated Exploratory Data Analysis (EDA) for tabular datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to explore. Must be non-empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from datacraft import AutoEDA
    >>> df = pd.DataFrame({"age": [25, 30, None, 45], "score": [85.0, 90.0, 88.0, 92.0]})
    >>> eda = AutoEDA(df)
    >>> info = eda.inspect()
    >>> stats = eda.summary()
    >>> qual = eda.quality()
    >>> eda.report()
    >>> eda.visualize()
    >>> eda.save_report("eda_report.html")
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"AutoEDA expects a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty (0 rows or 0 columns).")

        # Keep reference to the DataFrame. Never modify it.
        self._df = df
        self._n_rows, self._n_cols = df.shape
        self._col_types = AutoClean._detect_col_types(df)
        self._visualizations: Optional[dict[str, Any]] = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the analyzed DataFrame."""
        return self._df.copy()

    @property
    def col_types(self) -> dict[str, str]:
        """Return a dictionary mapping column names to detected data types."""
        return dict(self._col_types)

    # ------------------------------------------------------------------
    # 1. inspect()
    # ------------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        """
        Inspect dataset dimensions, column names, data types, missing values,
        unique value counts, and duplicate row count.

        Returns
        -------
        dict with keys:
          - 'shape': tuple (rows, columns)
          - 'rows': int
          - 'columns': list of column names
          - 'column_count': int
          - 'dtypes': dict of column -> raw pandas dtype string
          - 'col_types': dict of column -> detected type ('numerical', 'categorical', 'datetime')
          - 'missing': dict of column -> {'count': int, 'percentage': float}
          - 'total_missing': total number of missing cells
          - 'unique_counts': dict of column -> unique non-null values count
          - 'duplicates': count of duplicate rows
          - 'duplicate_percentage': percentage of duplicate rows
        """
        missing_dict: dict[str, dict[str, Any]] = {}
        unique_dict: dict[str, int] = {}

        for col in self._df.columns:
            s = self._df[col]
            missing_count = int(s.isnull().sum())
            missing_pct = float((missing_count / self._n_rows) * 100) if self._n_rows else 0.0
            missing_dict[col] = {
                "count": missing_count,
                "percentage": round(missing_pct, 2),
            }
            unique_dict[col] = int(s.nunique(dropna=True))

        total_missing = int(self._df.isnull().sum().sum())
        dup_count = int(self._df.duplicated().sum())

        return {
            "shape": (self._n_rows, self._n_cols),
            "rows": self._n_rows,
            "columns": list(self._df.columns),
            "column_count": self._n_cols,
            "dtypes": self._df.dtypes.astype(str).to_dict(),
            "col_types": dict(self._col_types),
            "missing": missing_dict,
            "total_missing": total_missing,
            "unique_counts": unique_dict,
            "duplicates": dup_count,
            "duplicate_percentage": round((dup_count / self._n_rows) * 100, 2) if self._n_rows else 0.0,
        }

    # ------------------------------------------------------------------
    # 2. summary()
    # ------------------------------------------------------------------

    def summary(self, column: Optional[str] = None) -> Union[dict[str, pd.DataFrame], pd.Series]:
        """
        Compute descriptive statistical summaries for numerical and categorical features.

        For numerical columns:
          - count, mean, median, standard deviation (std), minimum (min), maximum (max)
        For categorical columns:
          - unique (count of distinct values), top (most frequent value), freq (frequency of top value)

        Parameters
        ----------
        column : str, optional
            If provided, return the summary series for that specific column.

        Returns
        -------
        dict with:
          - 'numerical': pd.DataFrame with rows for each numerical column
          - 'categorical': pd.DataFrame with rows for each categorical column
        Or pd.Series if a single column is requested.
        """
        if column is not None:
            if column not in self._df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            ctype = self._col_types.get(column, "categorical")
            if ctype == "numerical":
                s = pd.to_numeric(self._df[column], errors="coerce").dropna()
                if not s.empty:
                    return pd.Series({
                        "count": len(s),
                        "mean": float(s.mean()),
                        "median": float(s.median()),
                        "std": float(s.std()) if len(s) > 1 else 0.0,
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }, name=column)
                return pd.Series({
                    "count": 0, "mean": np.nan, "median": np.nan,
                    "std": np.nan, "min": np.nan, "max": np.nan
                }, name=column)
            else:
                s = self._df[column].dropna()
                if not s.empty:
                    vc = s.value_counts()
                    return pd.Series({
                        "unique": int(s.nunique()),
                        "top": vc.index[0],
                        "freq": int(vc.iloc[0]),
                    }, name=column)
                return pd.Series({"unique": 0, "top": None, "freq": 0}, name=column)

        # Numerical summary DataFrame
        num_cols = [c for c, t in self._col_types.items() if t == "numerical"]
        num_records = []
        for c in num_cols:
            s = pd.to_numeric(self._df[c], errors="coerce").dropna()
            if not s.empty:
                cnt = len(s)
                mean_val = float(s.mean())
                med_val = float(s.median())
                std_val = float(s.std()) if cnt > 1 else 0.0
                min_val = float(s.min())
                max_val = float(s.max())
            else:
                cnt = 0
                mean_val = np.nan
                med_val = np.nan
                std_val = np.nan
                min_val = np.nan
                max_val = np.nan

            num_records.append({
                "column": c,
                "count": cnt,
                "mean": mean_val,
                "median": med_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            })

        if num_records:
            num_df = pd.DataFrame(num_records).set_index("column")
        else:
            num_df = pd.DataFrame(columns=["count", "mean", "median", "std", "min", "max"])

        # Categorical summary DataFrame
        cat_cols = [c for c, t in self._col_types.items() if t == "categorical"]
        cat_records = []
        for c in cat_cols:
            s = self._df[c].dropna()
            if not s.empty:
                n_unique = int(s.nunique())
                vc = s.value_counts()
                top_val = vc.index[0]
                top_freq = int(vc.iloc[0])
            else:
                n_unique = 0
                top_val = None
                top_freq = 0

            cat_records.append({
                "column": c,
                "unique": n_unique,
                "top": top_val,
                "freq": top_freq,
            })

        if cat_records:
            cat_df = pd.DataFrame(cat_records).set_index("column")
        else:
            cat_df = pd.DataFrame(columns=["unique", "top", "freq"])

        return {
            "numerical": num_df,
            "categorical": cat_df,
        }

    # ------------------------------------------------------------------
    # 3. correlations()
    # ------------------------------------------------------------------

    def correlations(self, threshold: float = 0.85, method: str = "pearson") -> dict[str, Any]:
        """
        Calculate numerical correlation matrix and identify highly correlated column pairs.
        Does not remove or modify any columns.

        Parameters
        ----------
        threshold : float, default 0.85
            Correlation coefficient magnitude threshold (|r| >= threshold) to flag.
        method : {'pearson', 'kendall', 'spearman'}, default 'pearson'
            Correlation method passed to pandas DataFrame.corr().

        Returns
        -------
        dict with:
          - 'matrix': pd.DataFrame correlation matrix
          - 'high_correlations': list of dicts with 'col1', 'col2', 'correlation', 'abs_correlation'
          - 'threshold': float
        """
        num_cols = [c for c, t in self._col_types.items() if t == "numerical"]
        if len(num_cols) < 2:
            empty_corr = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
            return {
                "matrix": empty_corr,
                "high_correlations": [],
                "threshold": threshold,
            }

        num_df = self._df[num_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = num_df.corr(method=method)

        high_corrs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val) and abs(val) >= threshold:
                    high_corrs.append({
                        "col1": c1,
                        "col2": c2,
                        "correlation": round(float(val), 4),
                        "abs_correlation": round(float(abs(val)), 4),
                    })

        high_corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return {
            "matrix": corr_matrix,
            "high_correlations": high_corrs,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # 4. quality()
    # ------------------------------------------------------------------

    def quality(self) -> dict[str, Any]:
        """
        Provide a consolidated data-quality summary:
          - missing-value percentage (overall and per column)
          - duplicate row count and percentage
          - constant columns (flagged, not dropped)
          - high-cardinality categorical columns (nunique >= 10)
          - numerical outlier counts and fence details (IQR method)

        Returns
        -------
        dict with quality indicators.
        """
        total_cells = self._n_rows * self._n_cols
        total_missing = int(self._df.isnull().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells else 0.0

        missing_by_col = {
            col: {
                "count": int(self._df[col].isnull().sum()),
                "percentage": round(float((self._df[col].isnull().sum() / self._n_rows) * 100), 2),
            }
            for col in self._df.columns
            if self._df[col].isnull().sum() > 0
        }

        dup_count = int(self._df.duplicated().sum())
        dup_pct = round((dup_count / self._n_rows) * 100, 2) if self._n_rows else 0.0

        # Constant columns (ignoring completely empty ones)
        constant_cols = [
            c for c in self._df.columns
            if not self._df[c].isnull().all() and self._df[c].nunique(dropna=True) <= 1
        ]

        # Empty columns (100% NaN)
        empty_cols = [c for c in self._df.columns if self._df[c].isnull().all()]

        # High cardinality categories (nunique >= 10)
        high_card_dict = AutoClean._detect_high_cardinality(self._df, self._col_types, threshold=10)
        high_card_cols = list(high_card_dict.keys())

        # Outliers using IQR fences (threshold 1.5)
        outliers_raw = AutoClean._detect_outliers(self._df, self._col_types, threshold=1.5)
        outlier_counts = {
            col: info["count"]
            for col, info in outliers_raw.items()
            if info["count"] > 0
        }
        total_outliers = sum(outlier_counts.values())

        return {
            "missing_percentage": missing_pct,
            "total_missing": total_missing,
            "missing_by_column": missing_by_col,
            "duplicate_count": dup_count,
            "duplicate_percentage": dup_pct,
            "constant_columns": constant_cols,
            "empty_columns": empty_cols,
            "high_cardinality_columns": high_card_cols,
            "high_cardinality": high_card_dict,
            "outlier_counts": outlier_counts,
            "total_outliers": total_outliers,
            "outlier_details": {
                col: info for col, info in outliers_raw.items() if info["count"] > 0
            },
        }

    # ------------------------------------------------------------------
    # 5. report()
    # ------------------------------------------------------------------

    def report(self, print_report: bool = True) -> str:
        """
        Print and return a clean, human-readable EDA report combining structural,
        statistical, quality, and correlation findings.

        Parameters
        ----------
        print_report : bool, default True
            Whether to print the report to stdout.

        Returns
        -------
        str : The full text report.
        """
        sep = "=" * 64
        sub_sep = "-" * 64

        insp = self.inspect()
        summ = self.summary()
        qual = self.quality()
        corr_info = self.correlations()

        lines = [
            sep,
            "  DataCraft -- AutoEDA Report",
            sep,
            f"  Dataset Shape   : {insp['rows']} rows x {insp['column_count']} columns",
            f"  Duplicate Rows  : {insp['duplicates']} ({insp['duplicate_percentage']}%)",
            f"  Total Missing   : {insp['total_missing']} cells ({qual['missing_percentage']}%)",
            "",
            "  Column Types Summary:",
        ]

        type_counts = Counter(self._col_types.values())
        for ctype, count in sorted(type_counts.items()):
            lines.append(f"    - {ctype:<15}: {count} column(s)")

        # Data Quality Section
        lines.extend([
            "",
            sub_sep,
            "  Data Quality Findings:",
            sub_sep,
        ])

        if qual["missing_by_column"]:
            lines.append("  * Missing Values:")
            for col, m in qual["missing_by_column"].items():
                lines.append(f"      {col:<26}: {m['count']:>4} missing ({m['percentage']}%)")
        else:
            lines.append("  * Missing Values: None detected (100% complete).")

        if qual["constant_columns"]:
            lines.append(f"  * Constant Columns ({len(qual['constant_columns'])}): {qual['constant_columns']}")
        else:
            lines.append("  * Constant Columns: None")

        if qual.get("empty_columns"):
            lines.append(f"  * Empty Columns ({len(qual['empty_columns'])}): {qual['empty_columns']}")

        if qual["high_cardinality_columns"]:
            lines.append(f"  * High-Cardinality Categoricals ({len(qual['high_cardinality_columns'])}):")
            for c in qual["high_cardinality_columns"]:
                lines.append(f"      {c:<26}: {qual['high_cardinality'][c]} unique values")
        else:
            lines.append("  * High-Cardinality Categoricals: None")

        if qual["outlier_counts"]:
            lines.append(f"  * Numerical Outliers (IQR fences, total {qual['total_outliers']}):")
            for c, cnt in qual["outlier_counts"].items():
                det = qual["outlier_details"][c]
                lines.append(f"      {c:<26}: {cnt:>3} outlier(s)  fence [{det['lower_fence']:.3g}, {det['upper_fence']:.3g}]")
        else:
            lines.append("  * Numerical Outliers: None detected")

        # Numerical Summary Section
        num_df = summ["numerical"]
        if not num_df.empty:
            lines.extend([
                "",
                sub_sep,
                "  Numerical Features Summary:",
                sub_sep,
            ])
            lines.append(f"  {'Column':<20} {'Count':>6} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            lines.append("  " + "-" * 80)
            for col, row in num_df.iterrows():
                lines.append(
                    f"  {col:<20} {row['count']:>6.0f} {row['mean']:>10.2f} {row['median']:>10.2f} "
                    f"{row['std']:>10.2f} {row['min']:>10.2f} {row['max']:>10.2f}"
                )

        # Categorical Summary Section
        cat_df = summ["categorical"]
        if not cat_df.empty:
            lines.extend([
                "",
                sub_sep,
                "  Categorical Features Summary:",
                sub_sep,
            ])
            lines.append(f"  {'Column':<20} {'Unique':>8} {'Top Value':<24} {'Freq':>8}")
            lines.append("  " + "-" * 64)
            for col, row in cat_df.iterrows():
                top_str = str(row['top']) if row['top'] is not None else "None"
                if len(top_str) > 22:
                    top_str = top_str[:19] + "..."
                lines.append(
                    f"  {col:<20} {row['unique']:>8} {top_str:<24} {row['freq']:>8}"
                )

        # Correlations Section
        high_corrs = corr_info["high_correlations"]
        if high_corrs:
            lines.extend([
                "",
                sub_sep,
                f"  High Correlations (|r| >= {corr_info['threshold']}):",
                sub_sep,
            ])
            for hc in high_corrs:
                lines.append(f"    * '{hc['col1']}' <-> '{hc['col2']}': r = {hc['correlation']:+.4f}")

        lines.extend([
            sep,
            "",
        ])

        report_str = "\n".join(lines)
        if print_report:
            print(report_str)
        return report_str

    # ------------------------------------------------------------------
    # 6. visualize()
    # ------------------------------------------------------------------

    def visualize(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        max_cols: int = 10,
    ) -> dict[str, Any]:
        """
        Generate exploratory plots automatically based on column types:
          - Numerical columns: histogram and boxplot
          - Categorical columns: top category frequency bar chart
          - Correlations: correlation heatmap (if at least 2 numerical columns)

        Visualization rules:
          - Automatically chooses suitable plots based on column type.
          - Skips empty/all-null columns gracefully.
          - Safely limits generated plots up to `max_cols` to avoid overwhelming large datasets.
          - Safely handles single-value and constant columns without crashing.
          - Strictly read-only: never modifies the caller's DataFrame.

        Parameters
        ----------
        output_dir : str or Path, optional
            If provided, saves generated plots as PNG files in this directory.
        max_cols : int, default 10
            Maximum number of columns of each type to plot.

        Returns
        -------
        dict with:
          - 'numerical': dict mapping column name -> {'histogram': ..., 'boxplot': ...}
          - 'categorical': dict mapping column name -> {'bar_chart': ...}
          - 'correlations': dict with {'heatmap': ...} (if >= 2 numerical columns)
          - 'saved_files': list of file paths saved to disk (if output_dir provided)
          - 'plots_count': total number of generated plots
        """
        saved_files: list[str] = []
        out_dir_str = str(output_dir) if output_dir is not None else None
        if out_dir_str:
            os.makedirs(out_dir_str, exist_ok=True)

        num_res: dict[str, dict[str, str]] = {}
        cat_res: dict[str, dict[str, str]] = {}
        corr_res: dict[str, str] = {}

        # --------------------------------------------------------------
        # 1. Numerical Columns: Histogram & Boxplot
        # --------------------------------------------------------------
        num_cols = [c for c, t in self._col_types.items() if t == "numerical"]
        valid_num_cols = [c for c in num_cols if not self._df[c].dropna().empty]
        target_num_cols = valid_num_cols[:max_cols]

        for col in target_num_cols:
            s = pd.to_numeric(self._df[col], errors="coerce").dropna()
            if s.empty:
                continue

            sanitized_col = _sanitize_name(col)
            col_plots: dict[str, str] = {}

            # --- Histogram ---
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            if s.nunique() <= 1:
                val = s.iloc[0]
                ax.bar([str(val)], [len(s)], color="#4f46e5", edgecolor="#312e81", alpha=0.85, width=0.4)
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram: {col} (Single Value: {val})", fontsize=11, fontweight="bold")
            else:
                bins = min(25, max(5, int(np.sqrt(len(s)))))
                ax.hist(s, bins=bins, color="#4f46e5", edgecolor="#312e81", alpha=0.85)
                mean_val = float(s.mean())
                med_val = float(s.median())
                ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
                ax.axvline(med_val, color="#10b981", linestyle="-", linewidth=1.5, label=f"Median: {med_val:.2f}")
                ax.legend(fontsize=9, loc="upper right")
                ax.set_title(f"Histogram: {col}", fontsize=11, fontweight="bold")
                ax.set_xlabel(str(col))
                ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            png_bytes = buf.read()
            b64_hist = base64.b64encode(png_bytes).decode("utf-8")

            if out_dir_str:
                hist_path = os.path.join(out_dir_str, f"hist_{sanitized_col}.png")
                with open(hist_path, "wb") as f:
                    f.write(png_bytes)
                saved_files.append(hist_path)
                col_plots["histogram"] = hist_path
            else:
                col_plots["histogram"] = b64_hist
            col_plots["histogram_base64"] = b64_hist

            # --- Boxplot ---
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            if s.nunique() <= 1:
                val = s.iloc[0]
                ax.plot([1], [val], "o", color="#06b6d4", markersize=8)
                ax.set_xticks([1])
                ax.set_xticklabels([str(col)])
                ax.set_title(f"Boxplot: {col} (Single Value: {val})", fontsize=11, fontweight="bold")
            else:
                bp = ax.boxplot(s, patch_artist=True, tick_labels=[str(col)], widths=0.4)
                for patch in bp.get("boxes", []):
                    patch.set_facecolor("#06b6d4")
                    patch.set_alpha(0.7)
                    patch.set_edgecolor("#0891b2")
                for median in bp.get("medians", []):
                    median.set_color("#e11d48")
                    median.set_linewidth(2)
                ax.set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
                ax.set_ylabel(str(col))
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            png_bytes = buf.read()
            b64_box = base64.b64encode(png_bytes).decode("utf-8")

            if out_dir_str:
                box_path = os.path.join(out_dir_str, f"box_{sanitized_col}.png")
                with open(box_path, "wb") as f:
                    f.write(png_bytes)
                saved_files.append(box_path)
                col_plots["boxplot"] = box_path
            else:
                col_plots["boxplot"] = b64_box
            col_plots["boxplot_base64"] = b64_box

            num_res[str(col)] = col_plots

        # --------------------------------------------------------------
        # 2. Categorical Columns: Top Category Frequency Bar Chart
        # --------------------------------------------------------------
        cat_cols = [c for c, t in self._col_types.items() if t == "categorical"]
        valid_cat_cols = [c for c in cat_cols if not self._df[c].dropna().empty]
        target_cat_cols = valid_cat_cols[:max_cols]

        for col in target_cat_cols:
            s = self._df[col].dropna().astype(str)
            if s.empty:
                continue

            vc = s.value_counts().head(10)
            if vc.empty:
                continue

            sanitized_col = _sanitize_name(col)
            col_plots = {}

            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            y_pos = np.arange(len(vc))
            bars = ax.barh(y_pos, vc.values, color="#10b981", edgecolor="#059669", alpha=0.85, height=0.6)
            ax.invert_yaxis()
            ax.set_yticks(y_pos)
            labels = [str(x)[:20] + "..." if len(str(x)) > 20 else str(x) for x in vc.index]
            ax.set_yticklabels(labels)
            max_val = max(vc.values) if len(vc) > 0 else 1
            for bar in bars:
                w = bar.get_width()
                offset = max(0.5, max_val * 0.02)
                ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{int(w)}",
                        va="center", ha="left", fontsize=9, color="#1f2937")
            ax.set_xlabel("Count")
            ax.set_title(f"Top Categories: {col}", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            png_bytes = buf.read()
            b64_bar = base64.b64encode(png_bytes).decode("utf-8")

            if out_dir_str:
                bar_path = os.path.join(out_dir_str, f"bar_{sanitized_col}.png")
                with open(bar_path, "wb") as f:
                    f.write(png_bytes)
                saved_files.append(bar_path)
                col_plots["bar_chart"] = bar_path
            else:
                col_plots["bar_chart"] = b64_bar
            col_plots["bar_chart_base64"] = b64_bar

            cat_res[str(col)] = col_plots

        # --------------------------------------------------------------
        # 3. Correlations: Correlation Heatmap
        # --------------------------------------------------------------
        if len(valid_num_cols) >= 2:
            heat_cols = valid_num_cols[:15]
            num_df = self._df[heat_cols].apply(pd.to_numeric, errors="coerce")
            corr = num_df.corr(method="pearson").fillna(0.0)

            if not corr.empty and corr.shape[0] >= 2 and corr.shape[1] >= 2:
                fig_dim = max(5.0, min(10.0, len(heat_cols) * 0.75 + 1.5))
                fig, ax = plt.subplots(figsize=(fig_dim, fig_dim * 0.85), dpi=100)
                sns.heatmap(
                    corr,
                    annot=(len(heat_cols) <= 12),
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1.0,
                    vmax=1.0,
                    center=0.0,
                    cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
                    ax=ax,
                    linewidths=0.5,
                )
                ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                plt.close(fig)
                buf.seek(0)
                png_bytes = buf.read()
                b64_heat = base64.b64encode(png_bytes).decode("utf-8")

                if out_dir_str:
                    heat_path = os.path.join(out_dir_str, "correlation_heatmap.png")
                    with open(heat_path, "wb") as f:
                        f.write(png_bytes)
                    saved_files.append(heat_path)
                    corr_res["heatmap"] = heat_path
                else:
                    corr_res["heatmap"] = b64_heat
                corr_res["heatmap_base64"] = b64_heat

        total_plots = sum(len([k for k in v if k in ("histogram", "boxplot")]) for v in num_res.values())
        total_plots += sum(len([k for k in v if k == "bar_chart"]) for v in cat_res.values())
        if "heatmap" in corr_res:
            total_plots += 1

        self._visualizations = {
            "numerical": num_res,
            "categorical": cat_res,
            "correlations": corr_res,
            "saved_files": saved_files,
            "plots_count": total_plots,
        }
        return self._visualizations

    # ------------------------------------------------------------------
    # 7. save_report()
    # ------------------------------------------------------------------

    def save_report(
        self,
        filepath: str = "eda_report.html",
        title: str = "DataCraft AutoEDA Report",
    ) -> str:
        """
        Generate and save a standalone, self-contained HTML EDA report.

        The report includes:
          - Dataset shape (rows, columns)
          - Column types (numerical, categorical, datetime)
          - Missing values and completion rates
          - Unique value counts
          - Numerical feature statistical summary
          - Categorical feature statistical summary
          - Correlations and high-correlation pairs
          - Consolidated data-quality warnings
          - Embedded visual plots (histograms, boxplots, category bars, correlation heatmap)

        Parameters
        ----------
        filepath : str, default "eda_report.html"
            Target path to write the standalone HTML file.
        title : str, default "DataCraft AutoEDA Report"
            Header title for the generated report.

        Returns
        -------
        str : Absolute path to the saved HTML file.
        """
        # Ensure visualizations are generated
        if self._visualizations is None:
            self.visualize()
        vis = self._visualizations or {"numerical": {}, "categorical": {}, "correlations": {}}

        insp = self.inspect()
        summ = self.summary()
        qual = self.quality()
        corr_info = self.correlations()

        # Build HTML content
        n_rows, n_cols = insp["rows"], insp["column_count"]
        total_missing = insp["total_missing"]
        missing_pct = qual["missing_percentage"]
        dup_count = insp["duplicates"]
        dup_pct = insp["duplicate_percentage"]
        outlier_total = qual["total_outliers"]

        # ---------------- Column Types & Schema Rows ----------------
        schema_rows_html = []
        for col in self._df.columns:
            ctype = self._col_types.get(col, "categorical")
            badge_cls = f"badge-{ctype}" if ctype in ("numerical", "categorical", "datetime") else "badge-numerical"
            raw_dtype = insp["dtypes"].get(col, "unknown")
            col_missing = insp["missing"].get(col, {"count": 0, "percentage": 0.0})
            m_cnt = col_missing["count"]
            m_pct = col_missing["percentage"]
            comp_pct = max(0.0, min(100.0, 100.0 - m_pct))
            u_cnt = insp["unique_counts"].get(col, 0)

            schema_rows_html.append(f"""
              <tr>
                <td><strong>{html.escape(str(col))}</strong></td>
                <td><span class="badge {badge_cls}">{html.escape(ctype)}</span></td>
                <td><code>{html.escape(str(raw_dtype))}</code></td>
                <td>{m_cnt:,} <span class="text-muted">({m_pct:.1f}%)</span></td>
                <td>
                  <div class="bar-container"><div class="bar-fill" style="width: {comp_pct:.1f}%;"></div></div>
                  <span>{comp_pct:.1f}%</span>
                </td>
                <td>{u_cnt:,}</td>
              </tr>
            """)

        # ---------------- Quality Warnings ----------------
        warnings_html = []
        if qual["missing_by_column"]:
            warn_items = "".join(
                f"<li><code>{html.escape(c)}</code>: {info['count']} missing ({info['percentage']}%)</li>"
                for c, info in qual["missing_by_column"].items()
            )
            warnings_html.append(f"""
              <div class="alert-box warning">
                <strong>Missing Values Detected:</strong> {len(qual['missing_by_column'])} column(s) have missing cells.
                <ul style="margin: 6px 0 0 18px;">{warn_items}</ul>
              </div>
            """)

        if dup_count > 0:
            warnings_html.append(f"""
              <div class="alert-box warning">
                <strong>Duplicate Rows:</strong> {dup_count:,} duplicate row(s) found ({dup_pct:.1f}% of total).
              </div>
            """)

        if qual["constant_columns"]:
            const_cols_str = ", ".join(f"<code>{html.escape(c)}</code>" for c in qual["constant_columns"])
            warnings_html.append(f"""
              <div class="alert-box warning">
                <strong>Constant Columns ({len(qual['constant_columns'])}):</strong> {const_cols_str} (single unique value).
              </div>
            """)

        if qual.get("empty_columns"):
            empty_cols_str = ", ".join(f"<code>{html.escape(c)}</code>" for c in qual["empty_columns"])
            warnings_html.append(f"""
              <div class="alert-box alert">
                <strong>Completely Empty Columns ({len(qual['empty_columns'])}):</strong> {empty_cols_str} (100% NaN).
              </div>
            """)

        if qual["high_cardinality_columns"]:
            card_items = "".join(
                f"<li><code>{html.escape(c)}</code>: {qual['high_cardinality'][c]} unique values</li>"
                for c in qual["high_cardinality_columns"]
            )
            warnings_html.append(f"""
              <div class="alert-box warning">
                <strong>High-Cardinality Categoricals:</strong>
                <ul style="margin: 6px 0 0 18px;">{card_items}</ul>
              </div>
            """)

        if qual["outlier_counts"]:
            outlier_items = "".join(
                f"<li><code>{html.escape(c)}</code>: {cnt} outlier(s) "
                f"[fence: {qual['outlier_details'][c]['lower_fence']:.3g}, {qual['outlier_details'][c]['upper_fence']:.3g}]</li>"
                for c, cnt in qual["outlier_counts"].items()
            )
            warnings_html.append(f"""
              <div class="alert-box warning">
                <strong>Numerical Outliers Detected (IQR method):</strong> {outlier_total:,} total outliers across {len(qual['outlier_counts'])} column(s).
                <ul style="margin: 6px 0 0 18px;">{outlier_items}</ul>
              </div>
            """)

        if not warnings_html:
            warnings_html.append("""
              <div class="alert-box success">
                <strong>Data Quality Healthy:</strong> No missing values, duplicate rows, constant columns, or severe outliers detected!
              </div>
            """)

        # ---------------- Numerical Summary Table ----------------
        num_summary_html = []
        num_df = summ["numerical"]
        if not num_df.empty:
            for c, r in num_df.iterrows():
                num_summary_html.append(f"""
                  <tr>
                    <td><strong>{html.escape(str(c))}</strong></td>
                    <td>{r['count']:.0f}</td>
                    <td>{r['mean']:.2f}</td>
                    <td>{r['median']:.2f}</td>
                    <td>{r['std']:.2f}</td>
                    <td>{r['min']:.2f}</td>
                    <td>{r['max']:.2f}</td>
                  </tr>
                """)
            num_section_body = f"""
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(num_summary_html)}
                </tbody>
              </table>
            """
        else:
            num_section_body = "<p class='empty-text'>No numerical columns detected in dataset.</p>"

        # ---------------- Categorical Summary Table ----------------
        cat_summary_html = []
        cat_df = summ["categorical"]
        if not cat_df.empty:
            for c, r in cat_df.iterrows():
                top_str = str(r['top']) if r['top'] is not None else "None"
                freq_pct = (r['freq'] / n_rows * 100) if n_rows else 0.0
                cat_summary_html.append(f"""
                  <tr>
                    <td><strong>{html.escape(str(c))}</strong></td>
                    <td>{r['unique']:,}</td>
                    <td><code>{html.escape(top_str)}</code></td>
                    <td>{r['freq']:,} <span class="text-muted">({freq_pct:.1f}%)</span></td>
                  </tr>
                """)
            cat_section_body = f"""
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Unique Values</th>
                    <th>Top Value</th>
                    <th>Frequency</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(cat_summary_html)}
                </tbody>
              </table>
            """
        else:
            cat_section_body = "<p class='empty-text'>No categorical columns detected in dataset.</p>"

        # ---------------- Correlations ----------------
        high_corrs = corr_info.get("high_correlations", [])
        corr_items_html = []
        if high_corrs:
            for hc in high_corrs:
                corr_items_html.append(
                    f"<li><code>{html.escape(hc['col1'])}</code> &harr; <code>{html.escape(hc['col2'])}</code>: "
                    f"<strong>r = {hc['correlation']:+.4f}</strong></li>"
                )
            corr_text = f"""
              <div class="alert-box warning" style="margin-bottom: 16px;">
                <strong>Highly Correlated Pairs (|r| &ge; {corr_info['threshold']}):</strong>
                <ul style="margin: 6px 0 0 18px;">{''.join(corr_items_html)}</ul>
              </div>
            """
        else:
            corr_text = "<p class='empty-text' style='margin-bottom: 16px;'>No correlation pairs exceeded threshold.</p>"

        corr_heatmap_b64 = vis.get("correlations", {}).get("heatmap_base64")
        if corr_heatmap_b64:
            heatmap_img_html = f"""
              <div class="plot-card" style="max-width: 650px; margin: 0 auto;">
                <h3>Correlation Heatmap</h3>
                <img src="data:image/png;base64,{corr_heatmap_b64}" alt="Correlation Heatmap" />
              </div>
            """
        else:
            heatmap_img_html = "<p class='empty-text'>Correlation heatmap requires at least 2 numerical features.</p>"

        # ---------------- Numerical Visualizations (Histogram + Boxplot) ----------------
        num_plots_html = []
        for col, pinfo in vis.get("numerical", {}).items():
            hist_b64 = pinfo.get("histogram_base64", "")
            box_b64 = pinfo.get("boxplot_base64", "")
            if hist_b64 and box_b64:
                num_plots_html.append(f"""
                  <div class="plot-card">
                    <h3>Distribution: {html.escape(col)}</h3>
                    <div class="plot-subgrid">
                      <img src="data:image/png;base64,{hist_b64}" alt="Histogram: {html.escape(col)}" />
                      <img src="data:image/png;base64,{box_b64}" alt="Boxplot: {html.escape(col)}" />
                    </div>
                  </div>
                """)

        # ---------------- Categorical Visualizations (Bar Charts) ----------------
        cat_plots_html = []
        for col, pinfo in vis.get("categorical", {}).items():
            bar_b64 = pinfo.get("bar_chart_base64", "")
            if bar_b64:
                cat_plots_html.append(f"""
                  <div class="plot-card">
                    <h3>Top Categories: {html.escape(col)}</h3>
                    <img src="data:image/png;base64,{bar_b64}" alt="Bar Chart: {html.escape(col)}" />
                  </div>
                """)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: #111827;
      --surface-elevated: #1f2937;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --border: #374151;
      --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #ef4444;
      --radius: 12px;
      --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 32px 20px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px 32px;
      margin-bottom: 24px;
      box-shadow: var(--shadow);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .header-title h1 {{
      font-size: 26px;
      font-weight: 700;
      background: var(--primary-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
    }}
    .header-title p {{ color: var(--text-muted); font-size: 13px; margin-top: 4px; }}
    .header-badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }}
    .kpi-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); }}
    .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; color: var(--text); }}
    .kpi-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
    .section-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow);
    }}
    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 18px;
    }}
    .section-header h2 {{ font-size: 18px; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th {{
      text-align: left;
      padding: 10px 14px;
      background: var(--surface-elevated);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
    }}
    td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .badge-numerical {{ background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }}
    .badge-categorical {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-datetime {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .bar-container {{
      width: 80px;
      height: 6px;
      background: var(--surface-elevated);
      border-radius: 3px;
      overflow: hidden;
      display: inline-block;
      vertical-align: middle;
      margin-right: 8px;
    }}
    .bar-fill {{ height: 100%; background: var(--accent-emerald); }}
    .alert-box {{
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 12px;
      font-size: 14px;
    }}
    .alert-box.success {{
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }}
    .alert-box.warning {{
      background: rgba(245, 158, 11, 0.08);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fbbf24;
    }}
    .alert-box.alert {{
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #f87171;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 20px;
    }}
    .plot-card {{
      background: var(--surface-elevated);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .plot-card h3 {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; color: var(--text-muted); text-align: left; }}
    .plot-card img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
    .plot-subgrid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .empty-text {{ color: var(--text-muted); font-size: 14px; font-style: italic; }}
    .text-muted {{ color: var(--text-muted); }}
    code {{ font-family: monospace; background: rgba(255, 255, 255, 0.05); padding: 2px 6px; border-radius: 4px; }}
    @media (max-width: 768px) {{
      .plot-subgrid {{ grid-template-columns: 1fr; }}
      body {{ padding: 16px 12px; }}
    }}
    footer {{ text-align: center; color: var(--text-muted); font-size: 13px; margin-top: 40px; padding-bottom: 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>{html.escape(title)}</h1>
        <p>Generated on {now_str} &bull; DataCraft Automated EDA</p>
      </div>
      <div class="header-badge">
        <span>{n_rows:,} rows &times; {n_cols:,} columns</span>
      </div>
    </header>

    <!-- Key Performance Indicators -->
    <div class="grid-kpi">
      <div class="kpi-card">
        <div class="kpi-label">Total Rows</div>
        <div class="kpi-value">{n_rows:,}</div>
        <div class="kpi-sub">Dataset records</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Columns</div>
        <div class="kpi-value">{n_cols:,}</div>
        <div class="kpi-sub">{len(qual.get('constant_columns', []))} constant</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Missing Values</div>
        <div class="kpi-value">{missing_pct:.1f}%</div>
        <div class="kpi-sub">{total_missing:,} missing cells</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Duplicate Rows</div>
        <div class="kpi-value">{dup_pct:.1f}%</div>
        <div class="kpi-sub">{dup_count:,} duplicate rows</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Outliers</div>
        <div class="kpi-value">{outlier_total:,}</div>
        <div class="kpi-sub">IQR fence detections</div>
      </div>
    </div>

    <!-- Data Quality Warnings -->
    <div class="section-card">
      <div class="section-header">
        <h2>Data-Quality Findings &amp; Warnings</h2>
      </div>
      {''.join(warnings_html)}
    </div>

    <!-- Column Types & Health Table -->
    <div class="section-card">
      <div class="section-header">
        <h2>Dataset Schema &amp; Column Types</h2>
      </div>
      <table>
        <thead>
          <tr>
            <th>Column Name</th>
            <th>Type</th>
            <th>DType</th>
            <th>Missing</th>
            <th>Completeness</th>
            <th>Unique Values</th>
          </tr>
        </thead>
        <tbody>
          {''.join(schema_rows_html)}
        </tbody>
      </table>
    </div>

    <!-- Numerical Features Summary -->
    <div class="section-card">
      <div class="section-header">
        <h2>Numerical Features Summary</h2>
      </div>
      {num_section_body}
    </div>

    <!-- Categorical Features Summary -->
    <div class="section-card">
      <div class="section-header">
        <h2>Categorical Features Summary</h2>
      </div>
      {cat_section_body}
    </div>

    <!-- Correlations -->
    <div class="section-card">
      <div class="section-header">
        <h2>Feature Correlations</h2>
      </div>
      {corr_text}
      {heatmap_img_html}
    </div>

    <!-- Generated Visualizations -->
    <div class="section-card">
      <div class="section-header">
        <h2>Numerical Distributions</h2>
      </div>
      {f'<div class="plot-grid">{"".join(num_plots_html)}</div>' if num_plots_html else '<p class="empty-text">No numerical distribution plots available.</p>'}
    </div>

    <div class="section-card">
      <div class="section-header">
        <h2>Categorical Distributions</h2>
      </div>
      {f'<div class="plot-grid">{"".join(cat_plots_html)}</div>' if cat_plots_html else '<p class="empty-text">No categorical distribution plots available.</p>'}
    </div>

    <footer>
      DataCraft AutoEDA &bull; Lightweight, transparent, deterministic tabular analysis
    </footer>
  </div>
</body>
</html>
"""
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_doc)

        return os.path.abspath(filepath)

