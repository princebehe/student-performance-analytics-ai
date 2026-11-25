"""
Unit tests for end-to-end inference and model prediction integrity.
"""
import pandas as pd
import numpy as np
import joblib
from src.config import MODELS_DIR, RISK_TIERS

MODEL_PATH = MODELS_DIR / "champion_pipeline.joblib"

def test_champion_model_file_exists():
    assert MODEL_PATH.exists(), f"Serialized model not found at {MODEL_PATH}"

def test_model_inference_output_contract():
    model = joblib.load(MODEL_PATH)
    sample_input = pd.DataFrame(
        {
            "age": [20],
            "study_hours_per_day": [3.5],
            "attendance_percentage": [75.0],
            "previous_score": [65.0],
            "assignment_score": [60.0],
            "midterm_score": [58.0],
            "sleep_hours": [7.0],
            "class_participation": [50.0],
            "gender": ["Female"],
            "extracurricular_activity": ["No"],
            "internet_access": ["Yes"],
            "parental_education": ["Bachelor"],
            "family_income_category": ["Medium"],
        }
    )

    prediction = model.predict(sample_input)
    probabilities = model.predict_proba(sample_input)

    assert len(prediction) == 1
    assert prediction[0] in RISK_TIERS
    assert probabilities.shape == (1, len(RISK_TIERS))
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-5)
