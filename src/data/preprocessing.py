"""
preprocessing.py

Text preprocessing utilities.

This module:
- Builds a single text field from multiple columns (e.g., title + description)
- Applies optional normalization (e.g., lowercasing)

Purpose:
- Provide a single, consistent way to construct model input text
"""
from __future__ import annotations
from typing import List
import pandas as pd

def build_text(df: pd.DataFrame, text_cols: List[str], sep: str = " ") -> pd.Series:
    parts = [df[c].fillna("").astype(str).str.strip() for c in text_cols]
    if not parts:
        return pd.Series([""] * len(df))
    out = parts[0]
    for p in parts[1:]:
        out = out + sep + p
    return out.str.strip()

def normalize_text(text: pd.Series, lowercase: bool = False) -> pd.Series:
    return text.str.lower() if lowercase else text