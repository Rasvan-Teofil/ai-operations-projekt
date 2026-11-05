"""predict_proba contract for both TF-IDF models."""

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


def test_tfidf_pipeline_is_built_in_features(monkeypatch):
    import src.sklearn_models as sklearn_models
    from src.features import build_tfidf_pipeline

    seen = []

    def spy(classifier):
        seen.append(type(classifier).__name__)
        return build_tfidf_pipeline(classifier)

    monkeypatch.setattr(sklearn_models, "build_tfidf_pipeline", spy)
    model = sklearn_models.new_logreg()
    assert seen == ["LogisticRegression"]
    assert set(model.pipeline.named_steps) == {"tfidf", "clf"}


def test_both_sklearn_models_share_the_proba_contract():
    frame = load_dataset()
    x_train, x_test, y_train, _ = split_dataset(frame)
    texts = x_train["text"].astype(str).tolist()
    labels = y_train.astype(str).tolist()
    for model in (new_logreg(), new_linearsvc()):
        model.fit(texts, labels)
        _assert_distribution(model, x_test["text"].astype(str).tolist())
