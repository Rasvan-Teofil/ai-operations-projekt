"""Der Standardlauf bleibt bei den TF-IDF-Modellen."""

import sys

from src.train import train


def test_default_train_does_not_import_transformers():
    before = set(sys.modules)
    rows = train()
    imported = set(sys.modules) - before
    assert {row["model"] for row in rows} == {"tfidf_logreg", "tfidf_linearsvc"}
    assert all("f1_macro" in row["metrics"] for row in rows)
    assert all(row["inference_ms_per_text"] >= 0 for row in rows)
    assert not any(name == "transformers" or name.startswith("transformers.") for name in imported)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
