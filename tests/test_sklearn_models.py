"""Vertrag predict_proba für die beiden TF-IDF-Modelle."""

from src.config import LABELS_3
from src.data import load_dataset, split_dataset
from src.sklearn_models import new_linearsvc, new_logreg


def _assert_distribution(model, texts):
    rows_in = list(texts)
    scores = model.predict_proba(rows_in)
    assert len(scores) == len(rows_in)
    for row in scores:
        assert set(row) == set(LABELS_3)
        assert abs(sum(row.values()) - 1.0) < 1e-5
    assert set(model.predict(rows_in)) <= set(LABELS_3)


def test_both_sklearn_models_share_the_proba_contract():
    frame = load_dataset()
    x_train, x_test, y_train, _ = split_dataset(frame)
    texts = x_train["text"].astype(str).tolist()
    labels = y_train.astype(str).tolist()
    for model in (new_logreg(), new_linearsvc()):
        model.fit(texts, labels)
        _assert_distribution(model, x_test["text"].astype(str).tolist())
