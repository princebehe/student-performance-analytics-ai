"""
Permutation importance and model interpretability module.
"""
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def compute_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Calculate permutation feature importance on holdout validation set."""
    perm_res = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="f1_weighted",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_importance = pd.DataFrame(
        {
            "Feature": X_val.columns,
            "Importance": perm_res.importances_mean,
            "Std": perm_res.importances_std,
        }
    ).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_importance

def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path | str,
    top_n: int = 10,
) -> None:
    """Render horizontal bar chart of top predictive features."""
    top_df = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(
        top_df["Feature"],
        top_df["Importance"],
        xerr=top_df["Std"],
        color="#1F4E79",
        alpha=0.85,
        capsize=4,
    )
    plt.title("Top Predictive Signals (Permutation Importance on Weighted F1)", fontsize=11, pad=10)
    plt.xlabel("Mean Performance Degradation (Weighted F1 Score)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
