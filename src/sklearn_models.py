"""TF-IDF models: baseline and classic alternative.

Both implement ``predict_proba``. LinearSVC has no real probabilities;
the decision function is turned into a distribution over the three
classes with a softmax. The argmax stays identical to LinearSVC's
``predict``.
"""

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    LINEARSVC_MAX_ITER,
    LOGREG_MAX_ITER,
    MODELS_DIR,
    RANDOM_SEED,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from src.interface import SentimentModel
from src.labels import complete_scores

MODEL_SPECS = {
    "tfidf_logreg": {
        "version": "tfidf-logreg-v1",
        "role": "baseline",
        "notes": "TF-IDF and logistic regression. Small, trainable on CPU, little semantics.",
    },
    "tfidf_linearsvc": {
        "version": "tfidf-linearsvc-v1",
        "role": "classic",
        "notes": "TF-IDF and LinearSVC. Similarly light, different decision boundary. Scores are a softmax of the decision function.",
    },
}


def _softmax(row: np.ndarray) -> np.ndarray:
    shifted = row - np.max(row)
    exp = np.exp(shifted)
    total = exp.sum()
    if total == 0:
        return np.full(row.shape, 1.0 / len(row))
    return exp / total


class SklearnSentimentModel(SentimentModel):
    def __init__(self, name: str, classifier):
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown sklearn model: {name}")
        spec = MODEL_SPECS[name]
        self.name = name
        self.model_version = spec["version"]
        self.role = spec["role"]
        self.notes = spec["notes"]
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF),
                ),
                ("clf", classifier),
            ]
        )
        self.artifact_path: Path | None = None
        self._probability_method = (
            "predict_proba" if hasattr(classifier, "predict_proba") else "softmax_decision_function"
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SklearnSentimentModel":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts: Sequence[str]) -> list[dict[str, float]]:
        rows = [str(text) for text in texts]
        class_names = [str(label) for label in self.pipeline.classes_]
        classifier = self.pipeline.named_steps["clf"]
        if hasattr(classifier, "predict_proba"):
            matrix = np.asarray(self.pipeline.predict_proba(rows), dtype=float)
        else:
            scores = np.atleast_2d(np.asarray(self.pipeline.decision_function(rows), dtype=float))
            if scores.shape[1] != len(class_names):
                raise RuntimeError("Decision function does not match the number of classes.")
            matrix = np.vstack([_softmax(row) for row in scores])
        distributions = []
        for probabilities in matrix:
            raw = {name: float(value) for name, value in zip(class_names, probabilities, strict=True)}
            distributions.append(complete_scores(raw))
        return distributions

    def save(self, directory: Path = MODELS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.joblib"
        joblib.dump(
            {
                "name": self.name,
                "model_version": self.model_version,
                "role": self.role,
                "backend": "sklearn",
                "estimator": self.pipeline,
                "probability_method": self._probability_method,
            },
            path,
        )
        self.artifact_path = path
        return path

    @classmethod
    def from_artifact(cls, path: Path) -> "SklearnSentimentModel":
        payload = joblib.load(path)
        model = cls.__new__(cls)
        model.name = payload["name"]
        model.model_version = payload["model_version"]
        model.role = payload.get("role", MODEL_SPECS[payload["name"]]["role"])
        model.notes = MODEL_SPECS[payload["name"]]["notes"]
        model.pipeline = payload["estimator"]
        model.artifact_path = path
        model._probability_method = payload.get("probability_method", "predict_proba")
        return model

    def tracking_params(self) -> dict[str, str | int | float | bool]:
        classifier = self.pipeline.named_steps["clf"]
        return {
            "vectorizer": "tfidf",
            "ngram_range": "1-2",
            "min_df": TFIDF_MIN_DF,
            "classifier": type(classifier).__name__,
            "probability_method": self._probability_method,
        }

    def size_mb(self) -> float | None:
        if self.artifact_path is not None and self.artifact_path.exists():
            return self.artifact_path.stat().st_size / (1024 * 1024)
        return None


def new_logreg() -> SklearnSentimentModel:
    classifier = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RANDOM_SEED)
    return SklearnSentimentModel("tfidf_logreg", classifier)


def new_linearsvc() -> SklearnSentimentModel:
    classifier = LinearSVC(max_iter=LINEARSVC_MAX_ITER, random_state=RANDOM_SEED, dual="auto")
    return SklearnSentimentModel("tfidf_linearsvc", classifier)


def train_sklearn_models(texts: Sequence[str], labels: Sequence[str]) -> list[SklearnSentimentModel]:
    """Trains baseline and LinearSVC on the same texts and saves both."""
    models = [new_logreg(), new_linearsvc()]
    for model in models:
        model.fit(texts, labels)
        model.save()
    return models


def load_sklearn_model(name: str, directory: Path = MODELS_DIR) -> SklearnSentimentModel:
    path = directory / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file missing: {path}. Please run `python -m src.train` first."
        )
    return SklearnSentimentModel.from_artifact(path)
