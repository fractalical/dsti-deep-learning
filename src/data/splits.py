"""
splits.py

Create and freeze a stratified train/validation split.

This module:
- Splits the original training dataset into train and validation sets
- Saves the split indices to a JSON file for reproducibility
- Writes processed train.csv and val.csv files

Purpose:
- Ensure reproducible and consistent data splits across all experiments
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def freeze_train_val_split(
    df_train: pd.DataFrame,
    out_split_json: Path,
    out_processed_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.10,
    label_col: str = "label",
) -> dict:
    """
    Create stratified train/val split from df_train and freeze indices to JSON.
    Also writes processed/train.csv and processed/val.csv.
    """
    if abs((train_ratio + val_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    idx = np.arange(len(df_train))
    y = df_train[label_col].values

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=y,
    )

    split_obj = {
        "meta": {
            "seed": seed,
            "strategy": "stratified_train_val",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "n_total": int(len(df_train)),
        },
        "splits": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        },
    }

    out_split_json.parent.mkdir(parents=True, exist_ok=True)
    out_processed_dir.mkdir(parents=True, exist_ok=True)

    out_split_json.write_text(json.dumps(split_obj, indent=2), encoding="utf-8")

    df_train.iloc[train_idx].to_csv(out_processed_dir / "train.csv", index=False)
    df_train.iloc[val_idx].to_csv(out_processed_dir / "val.csv", index=False)

    return split_obj

def load_frozen_split(split_json: Path) -> dict:
    if not split_json.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")
    return json.loads(split_json.read_text(encoding="utf-8"))