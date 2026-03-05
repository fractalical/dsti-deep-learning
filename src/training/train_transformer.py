from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.data.io import load_csv
from src.data.preprocessing import build_text, normalize_text
from src.models.transformer import TransformerFineTuner
from src.evaluation.metrics import compute_metrics
from src.training.utils import (
    load_yaml,
    make_run_dir,
    save_json,
    set_seed,
    set_determinism,
    snapshot_config,
    short_model_name,
)


def train_one(
    model_name: str,
    config_path: str,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    cfg = load_yaml(config_path)
    overrides = overrides or {}

    # ---- Repro
    seed = int(cfg["data"]["seed"])
    set_seed(seed)
    set_determinism(bool(cfg["data"].get("deterministic", False)))

    # ---- Paths
    train_csv = Path(cfg["paths"]["train_csv"])
    val_csv = Path(cfg["paths"]["val_csv"])
    runs_dir = Path(cfg["paths"]["runs_dir"])

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing processed files.\nExpected: {train_csv} and {val_csv}\n"
            f"Create them first (freeze split + processed CSVs)."
        )

    # ---- Load data
    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)

    # ---- Build text consistently from config
    text_cols = cfg["data"]["text_cols"]
    sep = cfg["data"]["text_sep"]
    lowercase = bool(cfg["data"].get("lowercase", False))

    train_text = normalize_text(build_text(train_df, text_cols=text_cols, sep=sep), lowercase=lowercase).tolist()
    val_text = normalize_text(build_text(val_df, text_cols=text_cols, sep=sep), lowercase=lowercase).tolist()

    train_y = train_df["label"].tolist()
    val_y = val_df["label"].tolist()

    # ---- Config params (+ overrides for ablations)
    num_labels = int(cfg["task"]["num_labels"])
    max_length = int(overrides.get("max_length", cfg["data"]["max_length"]))
    batch_size = int(overrides.get("batch_size", cfg["data"]["batch_size"]))

    tr = cfg["training"]
    epochs = int(overrides.get("epochs", tr["epochs"]))
    learning_rate = float(overrides.get("learning_rate", tr["learning_rate"]))
    weight_decay = float(overrides.get("weight_decay", tr["weight_decay"]))
    warmup_ratio = float(overrides.get("warmup_ratio", tr.get("warmup_ratio", 0.0)))
    max_grad_norm = float(overrides.get("max_grad_norm", tr.get("max_grad_norm", 1.0)))
    fp16 = bool(overrides.get("fp16", tr.get("fp16", False)))
    eval_strategy = str(overrides.get("eval_strategy", tr.get("eval_strategy", "epoch")))
    save_strategy = str(overrides.get("save_strategy", tr.get("save_strategy", "epoch")))
    logging_steps = int(overrides.get("logging_steps", tr.get("logging_steps", 50)))

    # ---- Run dir
    run_name = overrides.get("run_name")
    if run_name is None:
        run_name = short_model_name(model_name)
    run_dir = make_run_dir(runs_dir, run_name)
    snapshot_config(config_path, run_dir / "config_snapshot.yaml")

    # Save overrides for reproducibility
    save_json(overrides, run_dir / "overrides.json")

    # ---- Train
    print(f"Training transformer: {model_name}")
    print(f"Run dir: {run_dir}")
    print(f"Params: epochs={epochs}, lr={learning_rate}, max_length={max_length}, batch={batch_size}")

    tuner = TransformerFineTuner(model_name=model_name, num_labels=num_labels)
    outputs = tuner.train(
        train_texts=train_text,
        train_labels=train_y,
        val_texts=val_text,
        val_labels=val_y,
        output_dir=run_dir,
        seed=seed,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        fp16=fp16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
    )

    # ---- Predict on val + save metrics/preds
    pred_output = outputs.trainer.predict(outputs.trainer.eval_dataset)
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = np.argmax(logits, axis=-1).tolist()

    metrics = compute_metrics(val_y, pred_ids, set_name="val")
    save_json(metrics, run_dir / "metrics_val.json")

    out_pred = pd.DataFrame({"true_label": val_y, "pred_label": pred_ids, "text": val_text})
    out_pred.to_csv(run_dir / "predictions_val.csv", index=False)

    # Save training log history (loss curves etc.)
    try:
        (run_dir / "log_history.json").write_text(
            json.dumps(outputs.trainer.state.log_history, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    print(f"VAL accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"✓ Saved transformer run to: {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", nargs="?", default=None, help="e.g. distilbert-base-uncased or roberta-base")
    ap.add_argument("--config", default="configs/data.yaml", help="Path to YAML config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = args.model_name or cfg["transformer"]["model_name"]

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    train_one(model_name, args.config, overrides=overrides)


if __name__ == "__main__":
    main()