"""
io.py

Utility functions to load CSV datasets and verify that required columns
(label, title, description) are present.

Purpose:
- Centralize data loading
- Fail early if dataset schema is incorrect
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLS = ["label", "title", "description"]

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")

    return df