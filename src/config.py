"""Pfade, Seeds und der gemeinsame Drei-Klassen-Vertrag.

Training, Vergleich und API lesen dieselben Werte. ``data/sample`` bleibt
die Dummy-Stichprobe für Tests und CI. Der Standardlauf liest die
aufbereiteten Splits unter ``data/processed`` (nicht im Git).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "sentiment_sample.csv"
PREPARED_DIR = DATA_DIR / "processed"
PREPARED_TRAIN_PATH = PREPARED_DIR / "train.csv"
PREPARED_VAL_PATH = PREPARED_DIR / "val.csv"
PREPARED_TEST_PATH = PREPARED_DIR / "test.csv"
PREPARED_META_PATH = PREPARED_DIR / "metadata.json"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FINETUNED_DIR = MODELS_DIR / "finetuned"
FINETUNED_SMOKE_DIR = MODELS_DIR / "finetuned-smoke"

RANDOM_SEED = 42
TEST_SIZE = 0.2
# Echte Splits: 70 % Train, 15 % Val, 15 % Test, stratifiziert nach Label und Sprache.
PREPARE_TEST_SIZE = 0.15
PREPARE_VAL_SIZE = 0.15
MIN_TEXT_CHARS = 20

DATASET_ID_EN = "takala/financial_phrasebank"
DATASET_CONFIG_EN = "sentences_75agree"
DATASET_ID_DE = "Kenpache/multilingual-financial-sentiment"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
LANGUAGE_COLUMN = "language"
ORIGIN_COLUMN = "origin"

LABELS_3 = ("negative", "neutral", "positive")
ALLOWED_LANGUAGES = frozenset({"de", "en"})

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "article-sentiment"

# Praktische Wahl nach dem Vergleich auf dem echten Testsplit: höchstes macro-F1,
# dazu klein und schnell. Die CI setzt SENTIMENT_MODEL=tfidf_logreg und bleibt leicht.
DEFAULT_MODEL_NAME = "tfidf_linearsvc"

HF_MAX_TOKENS = 512
TABULARISAI_MODEL_ID = "tabularisai/multilingual-sentiment-analysis"
CARDIFFNLP_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_FINETUNE_BASE = "distilbert-base-multilingual-cased"

REQUEST_TIMEOUT_SECONDS = 15.0
USER_AGENT = "AIOpsSentimentBot/0.1 (HHN AI Operations coursework; single-article fetch)"

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
LOGREG_MAX_ITER = 1000
LINEARSVC_MAX_ITER = 2000


def selected_model_name() -> str:
    """Liest das Modell, das die API laden soll. Default: TF-IDF + LinearSVC."""
    raw = os.environ.get("SENTIMENT_MODEL", DEFAULT_MODEL_NAME).strip().lower()
    return raw or DEFAULT_MODEL_NAME
