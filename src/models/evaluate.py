"""
Model evaluation metrics and diagnostic visualizations.
"""
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    """Compute multi-class evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr": float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: Path | str,
    title: str = "Multi-Class Confusion Matrix",
) -> None:
    """Render and save normalized confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Academic Risk Tier", fontsize=10)
    plt.ylabel("Observed Academic Risk Tier", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
