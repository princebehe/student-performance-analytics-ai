"""
Central configuration for Student Performance Analytics System.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student_performance.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "student_performance_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

# Modeling Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature Taxonomy
TARGET_COLUMN = "academic_risk"
EXCLUDED_COLUMNS = ["student_id", "final_score"]

NUMERICAL_FEATURES = [
    "age",
    "study_hours_per_day",
    "attendance_percentage",
    "previous_score",
    "assignment_score",
    "midterm_score",
    "sleep_hours",
    "class_participation",
]

CATEGORICAL_FEATURES = [
    "gender",
    "extracurricular_activity",
    "internet_access",
    "parental_education",
    "family_income_category",
]

RISK_TIERS = ["Low Risk", "Moderate Risk", "High Risk"]
