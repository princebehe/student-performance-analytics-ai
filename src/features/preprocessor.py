"""
Feature transformation and leakage prevention module.
"""
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    TARGET_COLUMN,
    EXCLUDED_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target vector y.
    Strictly excludes database identifiers and post-outcome columns to prevent leakage.
    """
    drop_cols = [c for c in EXCLUDED_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]
    return X, y

def build_preprocessor() -> ColumnTransformer:
    """Build leak-free scikit-learn ColumnTransformer for numerical and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
