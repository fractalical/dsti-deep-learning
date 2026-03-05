from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def compute_metrics(y_true: List[int], y_pred: List[int], set_name: str = "val") -> Dict[str, Any]:
    """Compute Accuracy + Macro-F1 + confusion matrix + per-class report."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    return {
        "set": set_name,
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def hf_compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Hugging Face Trainer-compatible compute_metrics.
    Returns ONLY scalar metrics (Trainer logs these).
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]
    pred_ids = np.argmax(preds, axis=-1)

    acc = accuracy_score(labels, pred_ids)
    f1_macro = f1_score(labels, pred_ids, average="macro")

    return {"accuracy": float(acc), "macro_f1": float(f1_macro)}