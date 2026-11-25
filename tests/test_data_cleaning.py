"""
Unit tests for data cleaning and audit functionality.
"""
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def test_audit_identifies_duplicates_and_nulls():
    cleaner = DataCleaner()
    df_sample = pd.DataFrame(
        {
            "age": [18, 19, 18, np.nan],
            "study_hours_per_day": [2.5, 3.0, 2.5, 4.0],
            "gender": ["Female", "Male", "Female", None],
        }
    )

    audit = cleaner.audit(df_sample)
    assert audit["total_records"] == 4
    assert audit["duplicate_records"] == 1
    assert audit["total_missing_values"] == 2
    assert "age" in audit["missing_attributes"]
    assert "gender" in audit["missing_attributes"]

def test_clean_removes_duplicates_and_imputes():
    cleaner = DataCleaner()
    df_sample = pd.DataFrame(
        {
            "age": [20.0, 22.0, 20.0, np.nan],
            "study_hours_per_day": [3.0, 4.0, 3.0, np.nan],
            "attendance_percentage": [80.0, 90.0, 80.0, 70.0],
            "previous_score": [65.0, 75.0, 65.0, 60.0],
            "assignment_score": [70.0, 80.0, 70.0, 65.0],
            "midterm_score": [72.0, 82.0, 72.0, 68.0],
            "sleep_hours": [7.0, 8.0, 7.0, 6.5],
            "class_participation": [50.0, 60.0, 50.0, 55.0],
            "gender": ["Female", "Male", "Female", np.nan],
            "extracurricular_activity": ["Yes", "No", "Yes", "Yes"],
            "internet_access": ["Yes", "Yes", "Yes", "No"],
            "parental_education": ["Bachelor", "Master", "Bachelor", "Bachelor"],
            "family_income_category": ["Medium", "High", "Medium", "Low"],
        }
    )

    df_clean, audit = cleaner.clean(df_sample)
    assert len(df_clean) == 3  # Duplicate dropped
    assert df_clean.isnull().sum().sum() == 0  # No null values remain
