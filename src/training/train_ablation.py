from __future__ import annotations

import argparse

from src.training.train_transformer import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument(
        "--param",
        choices=["learning_rate", "max_length"],
        default="learning_rate",
        help="One factor to ablate",
    )
    ap.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="Space-separated values (e.g. 1e-5 2e-5) or (128 256)",
    )
    ap.add_argument("--epochs", type=int, default=1, help="Keep ablations small (default=1)")
    args = ap.parse_args()

    if args.param == "learning_rate":
        vals = [float(v) for v in args.values]
    else:
        vals = [int(v) for v in args.values]

    print(f"Ablation param={args.param} values={vals} model={args.model}")

    for v in vals:
        overrides = {"epochs": args.epochs, args.param: v, "run_name": f"ablation_{args.param}_{str(v).replace('.','p')}"}
        train_one(args.model, args.config, overrides=overrides)


if __name__ == "__main__":
    main()