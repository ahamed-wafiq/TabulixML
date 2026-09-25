"""
examples/basic_usage.py -- Demonstrates DataCraft.AutoClean on a synthetic dataset.

Run from the package root:
    python examples/basic_usage.py
"""

import numpy as np
import pandas as pd

from datacraft import AutoClean


# ---------------------------------------------------------------------------
# Build a messy synthetic DataFrame
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 20

_age_raw = rng.integers(18, 65, n).astype(float)
_age_mask = rng.random(n) < 0.15
_age = pd.Series(_age_raw)
_age[_age_mask] = np.nan

_salary_choices = [30000.0, 50000.0, 70000.0, 90000.0, 500000.0]
_salary_raw = pd.Series([_salary_choices[i] for i in rng.integers(0, len(_salary_choices), n)])
_salary_mask = rng.random(n) < 0.10
_salary_raw[_salary_mask] = np.nan

_dept_choices = ["Engineering", "Sales", "HR", "Engineering"]
_dept_raw = pd.Series([_dept_choices[i] for i in rng.integers(0, len(_dept_choices), n)], dtype=object)
_dept_mask = rng.random(n) < 0.10
_dept_raw[_dept_mask] = np.nan

raw_data = pd.DataFrame(
    {
        "Age": _age,
        "Annual Salary ($)": _salary_raw,
        "  Department  ": _dept_raw,
        "Hire Date": pd.to_datetime(
            np.random.choice(
                pd.date_range("2015-01-01", "2024-12-31", freq="ME"),
                n,
            )
        ),
        "always_zero": [0.0] * n,       # constant column
        "nothing": [np.nan] * n,        # completely empty column
    }
)

# Inject 3 duplicate rows
raw_data = pd.concat([raw_data, raw_data.iloc[[0, 1, 2]]], ignore_index=True)

print("=" * 60)
print(" RAW DATA (first 8 rows)")
print("=" * 60)
print(raw_data.head(8).to_string(index=False))
print(f"\nShape: {raw_data.shape}")

# ---------------------------------------------------------------------------
# Instantiate AutoClean
# ---------------------------------------------------------------------------

cleaner = AutoClean(
    raw_data,
    numerical_strategy="median",
    categorical_strategy="mode",
    remove_duplicates=True,
    clean_col_names=True,
    iqr_threshold=1.5,
)

# ---------------------------------------------------------------------------
# 1. Preview planned cleaning steps
# ---------------------------------------------------------------------------

print("\n")
cleaner.preview()

# ---------------------------------------------------------------------------
# 2. Full inspection report (before clean)
# ---------------------------------------------------------------------------

print("\n")
cleaner.report()

# ---------------------------------------------------------------------------
# 3. Clean
# ---------------------------------------------------------------------------

cleaned = cleaner.clean()

print("\n")
print("=" * 60)
print(" CLEANED DATA (first 8 rows)")
print("=" * 60)
print(cleaned.head(8).to_string(index=False))
print(f"\nShape after cleaning: {cleaned.shape}")

# ---------------------------------------------------------------------------
# 4. Report after clean (before/after comparison)
# ---------------------------------------------------------------------------

print("\n")
cleaner.report()

# ---------------------------------------------------------------------------
# 5. Cleaning history
# ---------------------------------------------------------------------------

print("\n")
print("=" * 60)
print(" CLEANING HISTORY")
print("=" * 60)
for i, entry in enumerate(cleaner.history(), 1):
    print(f"  [{i:02d}] {entry['timestamp']}  [{entry['operation']}]  {entry['message']}")
