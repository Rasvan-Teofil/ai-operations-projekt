"""Zentrale Pfade, Seeds und der Platzhalter-Datenvertrag.

Training und Inferenz lesen dieselben Werte, damit Split, Artefakt
und API nicht auseinanderlaufen. Sobald das echte Problem feststeht,
werden Spalten, Zielvariable und Pfade hier angepasst.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Wird von `python -m src.train` geschrieben und von der API gelesen.
MODEL_FILENAME = "baseline.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Eine Stelle für den Zufall: Split und Modellinitialisierung.
RANDOM_SEED = 42
TEST_SIZE = 0.2

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "ai-operations-baseline"

# PLACEHOLDER: Feature-Reihenfolge des Iris-Datensatzes aus scikit-learn.
# Dieselbe Reihenfolge nutzt der Input-Vertrag in app/schemas.py.
FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "species"
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
