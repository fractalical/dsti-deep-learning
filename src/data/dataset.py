"""
dataset.py

PyTorch Dataset for text classification.

This module:
- Takes a pandas DataFrame and a tokenizer
- Tokenizes text samples
- Returns input_ids, attention_mask, and labels

Purpose:
- Bridge between pandas data and PyTorch models
"""
from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        text = self.df.loc[i, "text"]
        label = int(self.df.loc[i, "label"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }