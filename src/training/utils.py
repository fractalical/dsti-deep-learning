from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to dict: {path}")
    return cfg


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def snapshot_config(config_path: str | Path, out_path: str | Path) -> None:
    config_path = Path(config_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_determinism(enabled: bool) -> None:
    """
    If enabled, try to make torch deterministic.
    This can fail for some ops; we keep it best-effort.
    """
    if not enabled:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def make_run_dir(runs_dir: str | Path, name: str) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def short_model_name(model_name: str) -> str:
    # "distilbert-base-uncased" -> "distilbert"
    # "FacebookAI/roberta-base" -> "roberta"
    last = model_name.split("/")[-1]
    return last.split("-")[0].lower()