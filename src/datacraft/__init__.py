"""
DataCraft -- Tabular ML Data Cleaning, Exploratory Analysis & Preprocessing Toolkit

Modules:
  - AutoClean : Automated, non-destructive tabular data cleaning and quality auditing.
  - AutoEDA   : Fast, read-only exploratory data analysis, summaries, and reporting.
  - AutoPrep  : Leak-free, reproducible ML preprocessing pipelines with scikit-learn.
  - AutoML    : Baseline model training, cross-validation, hyperparameter tuning, model persistence & inference.

Quick start:
>>> import pandas as pd
>>> from datacraft import AutoClean, AutoEDA, AutoPrep, AutoML
>>> df = pd.DataFrame({"a": [1, None, 2], "b": ["x", "y", "x"], "target": [0, 1, 0]})
>>> prep = AutoPrep(df, target="target")
>>> X_train, X_test, y_train, y_test = prep.prepare()
>>> automl = AutoML(df, target="target")
>>> automl.fit()
"""

from datacraft.cleaner import AutoClean
from datacraft.eda import AutoEDA
from datacraft.prep import AutoPrep
from datacraft.automl import AutoML

__version__ = "1.0.0"
__all__ = ["AutoClean", "AutoEDA", "AutoPrep", "AutoML"]
__author__ = "DataCraft contributors"

