"""Vergleich nutzt einen Testsplit, ohne Transformer zu laden."""

from src.config import TEXT_COLUMN
from src.compare import SELECTION_NOTE, run_comparison
from src.data import load_dataset, split_dataset
from src.interface import SentimentModel


class FakeModel(SentimentModel):
    def __init__(self, name: str):
        self.name = name
        self.model_version = f"{name}-v1"
        self.role = "test"
        self.notes = f"Notiz {name}"
        self.seen: list[str] | None = None

    def predict_proba(self, texts):
        self.seen = [str(text) for text in texts]
        return [{"negative": 0.2, "neutral": 0.3, "positive": 0.5} for _ in texts]


def test_comparison_scores_every_model_on_the_same_test_texts(tmp_path):
    first, second = FakeModel("alpha"), FakeModel("beta")
    rows = run_comparison(models=[first, second], output_dir=tmp_path, log_mlflow=False)

    frame = load_dataset()
    _, x_test, _, _ = split_dataset(frame)
    expected = x_test[TEXT_COLUMN].astype(str).tolist()
    assert first.seen == expected
    assert second.seen == expected
    assert {row["model"] for row in rows} == {"alpha", "beta"}
    assert "f1_macro" in rows[0]["metrics"]
    assert "f1_negative" in rows[0]["metrics"]
    assert rows[0]["confusion"].shape == (3, 3)

    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "macro-F1" in markdown
    assert "SENTIMENT_MODEL" in markdown
    assert SELECTION_NOTE.strip().splitlines()[0] in markdown
    assert (tmp_path / "comparison.csv").exists()
