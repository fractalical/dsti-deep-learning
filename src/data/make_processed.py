from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .io import load_csv
from .splits import freeze_train_val_split, load_frozen_split
from .checks import run_sanity_checks_train_val


def _read_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to a dict: {config_path}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml", help="Path to YAML config")
    ap.add_argument(
        "--force_new_split",
        action="store_true",
        help="Ignore existing split_json and recompute split (NOT recommended unless you must)",
    )
    args = ap.parse_args()

    cfg = _read_cfg(Path(args.config))

    # Expect the same structure used by configs/data.yaml
    paths = cfg["paths"]
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})

    raw_train_csv = Path(paths["raw_train_csv"])
    split_json = Path(paths["split_json"])
    processed_dir = Path(paths["processed_dir"])

    seed = int(data_cfg.get("seed", 42))
    train_ratio = float(split_cfg.get("train_ratio", 0.90))
    val_ratio = float(split_cfg.get("val_ratio", 0.10))
    stratify_col = str(split_cfg.get("stratify_col", "label"))

    print(f"Config: {args.config}")
    print(f"Raw train: {raw_train_csv}")
    print(f"Split json: {split_json}")
    print(f"Processed dir: {processed_dir}")

    # 1) Load raw data (validates required columns)
    df = load_csv(raw_train_csv)

    # 2) Create processed train/val using existing frozen split if present
    processed_dir.mkdir(parents=True, exist_ok=True)

    if split_json.exists() and not args.force_new_split:
        split_obj = load_frozen_split(split_json)
        train_idx = split_obj["splits"]["train"]
        val_idx = split_obj["splits"]["val"]

        (processed_dir / "train.csv").write_text(
            df.iloc[train_idx].to_csv(index=False), encoding="utf-8"
        )
        (processed_dir / "val.csv").write_text(
            df.iloc[val_idx].to_csv(index=False), encoding="utf-8"
        )
        print("✓ Reused existing frozen split JSON and wrote processed/train.csv + processed/val.csv")
    else:
        split_obj = freeze_train_val_split(
            df_train=df,
            out_split_json=split_json,
            out_processed_dir=processed_dir,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            label_col=stratify_col,
        )
        print("✓ Created new frozen split JSON and wrote processed/train.csv + processed/val.csv")

    # 3) Sanity checks (leakage, missing values, label distribution)
    report = run_sanity_checks_train_val(df, split_obj)
    sanity_path = processed_dir / "sanity_report.json"
    sanity_path.write_text(json.dumps(report.details, indent=2), encoding="utf-8")

    # 4) Quick label range check
    proc_train = pd.read_csv(processed_dir / "train.csv")
    labels = sorted(proc_train["label"].unique().tolist())
    print(f"Labels in processed train: {labels}")
    print(f"Sanity OK: {report.ok}")
    print(f"Saved sanity report: {sanity_path}")


if __name__ == "__main__":
    main()