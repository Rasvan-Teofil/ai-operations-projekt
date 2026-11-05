"""Fit the TF-IDF joblib artefacts that the API image serves.

No MLflow and no Transformer weights. The default reads the prepared
financial-sentence splits and downloads them first when they are missing
(``python -m src.prepare_data``). ``SENTIMENT_DATA=dummy`` uses the
committed 18-sentence sample and does not touch the network.

Run from the project root:

    python scripts/bake_sklearn_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABEL_COLUMN, PREPARED_TRAIN_PATH, TEXT_COLUMN  # noqa: E402
from src.data import data_mode, load_work_splits  # noqa: E402
from src.sklearn_models import train_sklearn_models  # noqa: E402


def main() -> None:
    mode = data_mode()
    if mode == "prepared" and not PREPARED_TRAIN_PATH.exists():
        from src.prepare_data import prepare

        prepare()
    train_frame, _val_frame, _test_frame = load_work_splits()
    models = train_sklearn_models(
        train_frame[TEXT_COLUMN].astype(str).tolist(),
        train_frame[LABEL_COLUMN].astype(str).tolist(),
    )
    for model in models:
        path = model.artifact_path
        if path is None or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Training did not write a model file for {model.name}.")
        print(f"saved {model.name} -> {path} ({path.stat().st_size} bytes) data={mode}")


if __name__ == "__main__":
    main()
