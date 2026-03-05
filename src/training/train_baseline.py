from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.io import load_csv
from src.data.preprocessing import build_text, normalize_text
from src.models.baseline import TFIDFLogRegBaseline
from src.training.utils import load_yaml, make_run_dir, save_json, set_seed, set_determinism, snapshot_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Paths
    train_csv = Path(cfg["paths"]["train_csv"])
    val_csv = Path(cfg["paths"]["val_csv"])
    runs_dir = Path(cfg["paths"]["runs_dir"])

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing processed files.\n"
            f"Expected: {train_csv} and {val_csv}\n"
            f"Create them first (freeze split + processed CSVs)."
        )

    # Repro
    seed = int(cfg["data"]["seed"])
    set_seed(seed)
    set_determinism(bool(cfg["data"].get("deterministic", False)))

    # Load processed data (schema checked)
    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)

    # Build consistent text field from config
    text_cols = cfg["data"]["text_cols"]
    sep = cfg["data"]["text_sep"]
    lowercase = bool(cfg["data"].get("lowercase", False))

    train_text = normalize_text(build_text(train_df, text_cols=text_cols, sep=sep), lowercase=lowercase).tolist()
    val_text = normalize_text(build_text(val_df, text_cols=text_cols, sep=sep), lowercase=lowercase).tolist()

    train_y = train_df["label"].tolist()
    val_y = val_df["label"].tolist()

    # Model
    baseline_cfg = cfg["baseline"]
    model = TFIDFLogRegBaseline(
        tfidf_cfg=baseline_cfg["tfidf"],
        logreg_cfg=baseline_cfg["logreg"],
        seed=seed,
    )

    # Train + eval
    print(f"Training baseline on {len(train_text)} samples, validating on {len(val_text)} samples...")
    model.fit(train_text, train_y)

    metrics, val_pred = model.evaluate(val_text, val_y, set_name="val")
    print(f"VAL accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    # Save run artifacts
    run_dir = make_run_dir(runs_dir, "baseline")
    snapshot_config(args.config, run_dir / "config_snapshot.yaml")

    model.save(run_dir)
    save_json(metrics, run_dir / "metrics_val.json")

    out_pred = pd.DataFrame({"true_label": val_y, "pred_label": val_pred, "text": val_text})
    out_pred.to_csv(run_dir / "predictions_val.csv", index=False)

    print(f"✓ Saved baseline run to: {run_dir}")


if __name__ == "__main__":
    main()