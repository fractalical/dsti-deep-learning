from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_metrics


class TFIDFLogRegBaseline:
    """
    Baseline: TF-IDF vectorizer + Logistic Regression.

    Important: YAML loads sequences as lists, but sklearn expects some params as tuples,
    e.g., ngram_range must be a tuple (1, 2), not [1, 2].
    """

    def __init__(self, tfidf_cfg: Dict[str, Any], logreg_cfg: Dict[str, Any], seed: int = 42):
        self.tfidf_cfg = dict(tfidf_cfg)
        self.logreg_cfg = dict(logreg_cfg)

        # ---- Fix YAML list -> sklearn tuple requirements
        if "ngram_range" in self.tfidf_cfg:
            ngr = self.tfidf_cfg["ngram_range"]
            if isinstance(ngr, list):
                if len(ngr) != 2:
                    raise ValueError(f"ngram_range must have 2 values, got: {ngr}")
                self.tfidf_cfg["ngram_range"] = (int(ngr[0]), int(ngr[1]))

        # ---- Build components
        self.vectorizer = TfidfVectorizer(**self.tfidf_cfg)

        # Ensure reproducibility
        self.model = LogisticRegression(**self.logreg_cfg, random_state=seed)

    def fit(self, texts: List[str], labels: List[int]) -> "TFIDFLogRegBaseline":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def evaluate(self, texts: List[str], labels: List[int], set_name: str = "val"):
        y_pred = self.predict(texts).tolist()
        metrics = compute_metrics(labels, y_pred, set_name=set_name)
        return metrics, y_pred

    def save(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "vectorizer.pkl").open("wb") as f:
            pickle.dump(self.vectorizer, f)
        with (run_dir / "model.pkl").open("wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, run_dir: str | Path) -> Tuple[TfidfVectorizer, LogisticRegression]:
        run_dir = Path(run_dir)
        with (run_dir / "vectorizer.pkl").open("rb") as f:
            vec = pickle.load(f)
        with (run_dir / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        return vec, model