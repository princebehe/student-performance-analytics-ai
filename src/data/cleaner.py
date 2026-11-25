"""
Data cleaning and quality assurance module.
"""
from typing import Dict, Any, Tuple
import pandas as pd
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

class DataCleaner:
    """Executes data quality audits, deduplication, and initial imputation."""

    def audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit dataset for duplicates and missing values."""
        null_counts = df.isnull().sum()
        missing_cols = null_counts[null_counts > 0].to_dict()
        return {
            "total_records": int(len(df)),
            "duplicate_records": int(df.duplicated().sum()),
            "total_missing_values": int(null_counts.sum()),
            "missing_attributes": missing_cols,
        }

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Deduplicate records and apply baseline imputation for analysis."""
        audit_initial = self.audit(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)

        # Median imputation for continuous attributes
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Mode imputation for categorical attributes
        for col in CATEGORICAL_FEATURES:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        audit_cleaned = self.audit(df_clean)
        return df_clean, {"initial": audit_initial, "cleaned": audit_cleaned}
