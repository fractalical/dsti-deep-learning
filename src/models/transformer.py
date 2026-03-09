from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.evaluation.metrics import hf_compute_metrics


class TextClsTorchDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        y = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


@dataclass
class TransformerRunOutputs:
    trainer: Trainer
    best_metric: Optional[float]


class TransformerFineTuner:
    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: Path,
        *,
        seed: int,
        max_length: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        max_grad_norm: float,
        fp16: bool,
        eval_strategy: str,
        save_strategy: str,
        logging_steps: int,
    ) -> TransformerRunOutputs:

        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = TextClsTorchDataset(train_texts, train_labels, self.tokenizer, max_length=max_length)
        val_ds = TextClsTorchDataset(val_texts, val_labels, self.tokenizer, max_length=max_length)

        # ---- Transformers compatibility:
        # Newer versions renamed `evaluation_strategy` -> `eval_strategy`.
        # We detect which name is accepted by inspecting the signature so that
        # unrelated TypeErrors are never swallowed by a broad except clause.
        _ta_params = inspect.signature(TrainingArguments.__init__).parameters
        _eval_strategy_kwarg = (
            "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
        )
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=int(epochs),
            per_device_train_batch_size=int(batch_size),
            per_device_eval_batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            warmup_ratio=float(warmup_ratio),
            max_grad_norm=float(max_grad_norm),
            fp16=bool(fp16),
            **{_eval_strategy_kwarg: eval_strategy},
            save_strategy=save_strategy,
            logging_steps=int(logging_steps),
            seed=int(seed),
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
        )

        # ---- Trainer API compatibility:
        # Newer versions accept `processing_class=...`, older versions accept `tokenizer=...`.
        # We detect which name is accepted by inspecting the signature.
        _trainer_params = inspect.signature(Trainer.__init__).parameters
        _tokenizer_kwarg = (
            "processing_class" if "processing_class" in _trainer_params else "tokenizer"
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            **{_tokenizer_kwarg: self.tokenizer},
            compute_metrics=hf_compute_metrics,
        )

        trainer.train()

        # Save "final_model" (best checkpoint loaded if load_best_model_at_end=True)
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        best_metric = None
        try:
            best_metric = trainer.state.best_metric
        except Exception:
            pass

        return TransformerRunOutputs(trainer=trainer, best_metric=best_metric)