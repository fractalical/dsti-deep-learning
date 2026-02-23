"""
checks.py

Run sanity checks on the dataset and the frozen split.

Checks include:
- Number of samples in each split
- Missing values
- Label distribution consistency
- Duplicate samples
- Potential data leakage between train and validation

Purpose:
- Detect common data issues before training
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd

@dataclass
class SanityReport:
    ok: bool
    details: Dict[str, Any]

def _exact_key(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str).str.strip()
    desc = df["description"].fillna("").astype(str).str.strip()
    return title + "||" + desc

def run_sanity_checks_train_val(df: pd.DataFrame, split_obj: dict) -> SanityReport:
    train_idx: List[int] = split_obj["splits"]["train"]
    val_idx: List[int] = split_obj["splits"]["val"]

    details: Dict[str, Any] = {}

   
    details["n_total"] = int(len(df))
    details["n_train"] = int(len(train_idx))
    details["n_val"] = int(len(val_idx))

    
    details["missing"] = df[["label", "title", "description"]].isna().sum().to_dict()

  
    def dist(idxs: List[int]) -> Dict[int, int]:
        return df.iloc[idxs]["label"].value_counts().sort_index().to_dict()

    details["label_dist"] = {
        "total": df["label"].value_counts().sort_index().to_dict(),
        "train": dist(train_idx),
        "val": dist(val_idx),
    }

   
    key = _exact_key(df)
    details["duplicate_rows_total"] = int(key.duplicated(keep=False).sum())

    train_set = set(key.iloc[train_idx].tolist())
    val_set = set(key.iloc[val_idx].tolist())
    details["leakage_exact_train_val"] = int(len(train_set & val_set))

   
    titles = df["title"].fillna("").astype(str).str.strip()
    ttrain = set(titles.iloc[train_idx].tolist())
    tval = set(titles.iloc[val_idx].tolist())
    details["leakage_same_title_train_val"] = int(len(ttrain & tval))

  
    ok = (
        details["n_train"] > 0
        and details["n_val"] > 0
        and all(int(v) == 0 for v in details["missing"].values())
    )

    return SanityReport(ok=ok, details=details)