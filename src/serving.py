"""One artifact loader shared by the comparison runner and HTTP service."""
import json
from pathlib import Path

import numpy as np

from src.config import LABELS


class ReferencePredictor:
    def __init__(self, path):
        import joblib
        import sklearn
        self.meta = json.loads((path / 'artifact.json').read_text())
        if self.meta['labels'] != LABELS or self.meta['sklearn'] != sklearn.__version__:
            raise ValueError('Reference artifact label order or sklearn version differs')
        payload = joblib.load(path / 'tfidf_linearsvc.joblib')
        self.pipeline = payload['estimator']
        self.version = payload['model_version']

    def predict_proba(self, texts):
        scores = self.pipeline.decision_function(list(texts))
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        order = [list(self.pipeline.classes_).index(label) for label in LABELS]
        return probabilities[:, order]


def load_artifact(path):
    path = Path(path)
    meta = json.loads((path / 'artifact.json').read_text())
    if meta['kind'] == 'local_llm':
        from src.local_llm import LocalLLMPredictor
        return LocalLLMPredictor(path)
    if meta['kind'] == 'reference_sklearn':
        return ReferencePredictor(path)
    from src.inference import Predictor
    return Predictor(path)
