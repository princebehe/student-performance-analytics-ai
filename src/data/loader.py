"""
Data ingestion and loading module.
"""
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_PATH

def load_dataset(filepath: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw student performance dataset."""
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset not found at {target_path}")
    return pd.read_csv(target_path)
