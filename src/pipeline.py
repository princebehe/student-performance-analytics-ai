"""
Central pipeline orchestrator for Student Performance Analytics System.
"""
import os
import json
import numpy as np
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    METRICS_PATH,
    FIGURES_DIR,
    RISK_TIERS,
)
from src.data.loader import load_dataset
from src.data.cleaner import DataCleaner
from src.models.train import ModelTrainer
from src.models.evaluate import plot_confusion_matrix
from src.models.explain import compute_permutation_importance, plot_feature_importance

def run_end_to_end_pipeline():
    """Execute complete ingestion, cleaning, benchmarking, and interpretability workflow."""
    print("1. Ingesting raw data...")
    df_raw = load_dataset(RAW_DATA_PATH)
    cleaner = DataCleaner()
    audit_summary = cleaner.audit(df_raw)
    print(f"   Loaded {audit_summary['total_records']} records ({audit_summary['duplicate_records']} duplicates, {audit_summary['total_missing_values']} missing values).")

    print("2. Sanitizing dataset...")
    df_clean, cleaning_audit = cleaner.clean(df_raw)
    os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"   Cleaned cohort saved to {PROCESSED_DATA_PATH} (Shape: {df_clean.shape}).")

    print("3. Training and benchmarking classifiers...")
    trainer = ModelTrainer()
    summary, champion_pipeline, (X_train, X_test, y_train, y_test) = trainer.train_and_evaluate(df_clean)
    champion_name = summary["champion_model"]
    champion_f1 = summary["champion_f1"]
    print(f"   Champion model selected: {champion_name} (Weighted F1: {champion_f1:.4f}).")

    print("4. Generating diagnostic figures...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    champion_metrics = summary["benchmark_results"][champion_name]
    cm = np.array(champion_metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm=cm,
        labels=RISK_TIERS,
        output_path=FIGURES_DIR / "fig5_confusion_matrix.png",
        title=f"Confusion Matrix - {champion_name}",
    )

    print("5. Evaluating permutation feature importance...")
    importance_df = compute_permutation_importance(
        model=champion_pipeline,
        X_val=X_test,
        y_val=y_test,
    )
    plot_feature_importance(
        importance_df=importance_df,
        output_path=FIGURES_DIR / "fig6_feature_importance.png",
    )

    print("6. Persisting run metrics summary...")
    output_payload = {
        "audit": cleaning_audit,
        "champion_model": champion_name,
        "champion_f1": champion_f1,
        "benchmark_results": summary["benchmark_results"],
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Pipeline executed successfully. Artifacts stored in {FIGURES_DIR.parent}.")

if __name__ == "__main__":
    run_end_to_end_pipeline()
