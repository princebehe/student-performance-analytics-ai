"""
Model training, benchmarking, and persistence module.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, RISK_TIERS, MODELS_DIR
from src.features.preprocessor import build_preprocessor, split_features_target
from src.models.evaluate import evaluate_classification

class ModelTrainer:
    """Trains and benchmarks multi-class classifiers using leak-free preprocessing pipelines."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=5,
                random_state=self.random_state,
            ),
        }

    def train_and_evaluate(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Any, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """Fit models, benchmark holdout metrics, and return the champion pipeline."""
        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=y,
        )

        results = {}
        best_model_name = None
        best_f1 = -1.0
        best_pipeline = None

        for name, clf in self.models.items():
            preprocessor = build_preprocessor()
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)

            metrics = evaluate_classification(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                labels=RISK_TIERS,
            )
            results[name] = metrics

            if metrics["weighted_f1"] > best_f1:
                best_f1 = metrics["weighted_f1"]
                best_model_name = name
                best_pipeline = pipeline

        summary = {
            "benchmark_results": results,
            "champion_model": best_model_name,
            "champion_f1": best_f1,
        }

        # Save champion model artifact
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_pipeline, MODELS_DIR / "champion_pipeline.joblib")

        return summary, best_pipeline, (X_train, X_test, y_train, y_test)
