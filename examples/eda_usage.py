"""
examples/eda_usage.py -- AutoEDA Usage Demonstration for TabulixML

Demonstrates the read-only exploratory data analysis module (AutoEDA):
  1. inspect()     : Dimensions, data types, missing values, duplicates
  2. summary()     : Numerical statistics (mean, median, std, min, max) and
                     categorical statistics (unique, top, freq)
  3. correlations(): Numerical correlation matrix & high correlation pairs
  4. quality()     : Consolidated quality audit (missing rates, duplicates, constant cols, outliers)
  5. report()      : Formatted terminal EDA report
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tabulixml import AutoEDA


def main():
    # Create a realistic sample dataset
    data = {
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "Alice"],
        "Age": [25.0, 30.0, np.nan, 45.0, 29.0, 38.0, 25.0],
        "Salary": [50000.0, 60000.0, 55000.0, 2500000.0, 52000.0, 75000.0, 50000.0],
        "Department": ["Engineering", "Sales", "HR", "Sales", "Engineering", "Marketing", "Engineering"],
        "Tenure_Years": [2.0, 4.0, 1.0, 12.0, 3.0, 6.0, 2.0],
        "Bonus_Pct": [0.10, 0.15, 0.05, 0.25, 0.10, 0.18, 0.10],
        "Status": ["Active"] * 7,  # Constant column
    }
    df = pd.DataFrame(data)

    print("=" * 70)
    print("  TabulixML AutoEDA Demonstration")
    print("=" * 70)
    print("\nOriginal DataFrame (first 5 rows):")
    print(df.head())

    # Initialize AutoEDA (strictly read-only)
    eda = AutoEDA(df)

    # 1. Inspect
    print("\n" + "-" * 70)
    print("1. inspect() - Structural Dataset Overview")
    print("-" * 70)
    inspection = eda.inspect()
    print(f"Shape               : {inspection['shape']}")
    print(f"Total Missing Cells : {inspection['total_missing']}")
    print(f"Duplicate Rows      : {inspection['duplicates']} ({inspection['duplicate_percentage']}%)")
    print(f"Detected Column Types: {inspection['col_types']}")

    # 2. Summary
    print("\n" + "-" * 70)
    print("2. summary() - Statistical Summaries")
    print("-" * 70)
    stats = eda.summary()
    print("\nNumerical Features Summary:")
    print(stats["numerical"])
    print("\nCategorical Features Summary:")
    print(stats["categorical"])

    # Single column summary example
    print("\nSingle Column Summary for 'Age':")
    print(eda.summary("Age"))

    # 3. Correlations
    print("\n" + "-" * 70)
    print("3. correlations() - Correlation Analysis (|r| >= 0.85)")
    print("-" * 70)
    corr_res = eda.correlations(threshold=0.85)
    print("Numerical Correlation Matrix:")
    print(corr_res["matrix"].round(3))
    print(f"\nHighly Correlated Pairs (|r| >= {corr_res['threshold']}):")
    for pair in corr_res["high_correlations"]:
        print(f"  * '{pair['col1']}' <-> '{pair['col2']}': r = {pair['correlation']:+.4f}")

    # 4. Quality
    print("\n" + "-" * 70)
    print("4. quality() - Data Quality Indicators")
    print("-" * 70)
    qual = eda.quality()
    print(f"Overall Missing Rate : {qual['missing_percentage']}%")
    print(f"Constant Columns     : {qual['constant_columns']}")
    print(f"Numerical Outliers   : {qual['outlier_counts']}")

    # 5. Full Report
    print("\n" + "-" * 70)
    print("5. report() - Complete Formatted Report")
    print("-" * 70)
    eda.report()

    # 6. Automatic Visualizations
    print("\n" + "-" * 70)
    print("6. visualize() - Automatic Visualizations")
    print("-" * 70)
    plots = eda.visualize(output_dir="eda_output")
    print(f"Generated {plots['plots_count']} plots in 'eda_output/' directory:")
    for f in plots["saved_files"]:
        print(f"  * {f}")

    # 7. Standalone HTML Report Export
    print("\n" + "-" * 70)
    print("7. save_report() - Standalone HTML Report")
    print("-" * 70)
    report_path = eda.save_report("eda_report.html")
    print(f"Standalone HTML report saved to: {report_path}")

    # Guarantee: original DataFrame is untouched
    print("\nVerification: Original DataFrame shape is still", df.shape)


if __name__ == "__main__":
    main()

