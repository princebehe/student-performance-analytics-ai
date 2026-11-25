"""
Models package exports.
"""
from src.models.train import ModelTrainer
from src.models.evaluate import evaluate_classification, plot_confusion_matrix
from src.models.explain import compute_permutation_importance, plot_feature_importance

__all__ = [
    "ModelTrainer",
    "evaluate_classification",
    "plot_confusion_matrix",
    "compute_permutation_importance",
    "plot_feature_importance",
]
