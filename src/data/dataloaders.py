"""
dataloaders.py

Create tokenizer and PyTorch DataLoaders for train and validation sets.

This module:
- Loads processed CSV files
- Applies text preprocessing
- Builds Dataset objects
- Wraps them into DataLoaders

Purpose:
- Provide ready-to-use train_loader and val_loader
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .io import load_csv
from .preprocessing import build_text, normalize_text
from .dataset import TextClsDataset

def prepare_df(df: pd.DataFrame, text_cols, sep: str, lowercase: bool) -> pd.DataFrame:
    df = df.copy()
    df["text"] = build_text(df, list(text_cols), sep=sep)
    df["text"] = normalize_text(df["text"], lowercase=lowercase)
    return df

def make_loaders(
    train_csv: Path,
    val_csv: Path,
    tokenizer_name: str,
    text_cols=("title", "description"),
    sep: str = " ",
    lowercase: bool = False,
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 2,
    shuffle_train: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    train_df = prepare_df(load_csv(train_csv), text_cols, sep, lowercase)
    val_df = prepare_df(load_csv(val_csv), text_cols, sep, lowercase)

    train_ds = TextClsDataset(train_df, tokenizer, max_length=max_length)
    val_ds = TextClsDataset(val_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return tokenizer, train_loader, val_loader