"""
Unit tests for feature splitting, preprocessor, and pipeline leakage checks.
"""
import pandas as pd
import numpy as np
from src.features.preprocessor import split_features_target, build_preprocessor
from src.config import TARGET_COLUMN, EXCLUDED_COLUMNS

def test_split_features_target_prevents_leakage():
    df_dummy = pd.DataFrame(
        {
            "student_id": ["STU001", "STU002"],
            "age": [19, 20],
            "final_score": [75.0, 85.0],
            "academic_risk": ["Low Risk", "Moderate Risk"],
        }
    )

    X, y = split_features_target(df_dummy)

    # Asserts that student_id and final_score are eliminated from X
    for col in EXCLUDED_COLUMNS:
        assert col not in X.columns, f"Data leakage detected: {col} present in feature matrix!"
    assert TARGET_COLUMN not in X.columns
    assert len(y) == 2
    assert list(y) == ["Low Risk", "Moderate Risk"]

def test_preprocessor_transformation_dimensions():
    preprocessor = build_preprocessor()
    df_sample = pd.DataFrame(
        {
            "age": [18, 21],
            "study_hours_per_day": [2.0, 4.5],
            "attendance_percentage": [75.0, 88.0],
            "previous_score": [60.0, 80.0],
            "assignment_score": [65.0, 85.0],
            "midterm_score": [70.0, 90.0],
            "sleep_hours": [7.0, 8.0],
            "class_participation": [50.0, 75.0],
            "gender": ["Female", "Male"],
            "extracurricular_activity": ["Yes", "No"],
            "internet_access": ["Yes", "Yes"],
            "parental_education": ["Bachelor", "Master"],
            "family_income_category": ["Medium", "High"],
        }
    )

    transformed = preprocessor.fit_transform(df_sample)
    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()
