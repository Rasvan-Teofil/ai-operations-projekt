"""Lädt die beiden öffentlichen Datensätze und schreibt Train/Val/Test.

Vom Projektroot:

    python -m src.prepare_data

Die Rohtexte bleiben unter ``data/processed/`` und sind gitignoriert.
Die Dummy-Datei unter ``data/sample/`` wird nicht angefasst.

Englisch: ``takala/financial_phrasebank``, Datei
``Sentences_75Agree.txt`` aus ``data/FinancialPhraseBank-v1.0.zip``.
``datasets.load_dataset`` scheitert an diesem Stand, weil das Repo noch
ein Legacy-Script enthält. Die Zip-Datei ist die veröffentlichte Quelle.

Deutsch: ``Kenpache/multilingual-financial-sentiment``, nur ``language=de``.

Split, einmal, für alle Modelle: stratifiziert nach ``label|language``,
Seed 42. Zuerst 15 % Test, aus dem Rest dann Val so, dass Val 15 % der
Gesamtheit ist (15/85 des Rests). Train sind die übrigen 70 %.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALLOWED_LANGUAGES,
    DATASET_CONFIG_EN,
    DATASET_ID_DE,
    DATASET_ID_EN,
    LABEL_COLUMN,
    LABELS_3,
    LANGUAGE_COLUMN,
    MIN_TEXT_CHARS,
    PREPARE_TEST_SIZE,
    PREPARE_VAL_SIZE,
    PREPARED_DIR,
    PREPARED_META_PATH,
    PREPARED_TEST_PATH,
    PREPARED_TRAIN_PATH,
    PREPARED_VAL_PATH,
    RANDOM_SEED,
    TEXT_COLUMN,
)

PHRASEBANK_ZIP = "data/FinancialPhraseBank-v1.0.zip"
PHRASEBANK_MEMBER = "FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
KENPACHE_CSV = "all_languages_clean.csv"
OUTPUT_COLUMNS = [TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN, "source", "dataset"]


def stratified_train_val_test(
    frame: pd.DataFrame,
    seed: int = RANDOM_SEED,
    test_size: float = PREPARE_TEST_SIZE,
    val_size: float = PREPARE_VAL_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15, stratifiziert nach Label und Sprache.

    Jede Schicht ``label|language`` muss mindestens zwei Zeilen haben,
    sonst kann sklearn nicht in beide Teile stratifizieren.
    """
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size und val_size müssen positiv sein und zusammen unter 1 liegen.")
    work = frame.reset_index(drop=True).copy()
    work["_stratum"] = work[LABEL_COLUMN].astype(str) + "|" + work[LANGUAGE_COLUMN].astype(str)
    counts = work["_stratum"].value_counts()
    tiny = counts[counts < 2]
    if not tiny.empty:
        raise ValueError(
            "Zu wenige Zeilen für eine stratifizierte Drei-Teilung: "
            + ", ".join(f"{name}={int(count)}" for name, count in tiny.items())
        )
    rest, test = train_test_split(
        work,
        test_size=test_size,
        random_state=seed,
        stratify=work["_stratum"],
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        rest,
        test_size=relative_val,
        random_state=seed,
        stratify=rest["_stratum"],
    )
    return _without_stratum(train), _without_stratum(val), _without_stratum(test)


def _without_stratum(part: pd.DataFrame) -> pd.DataFrame:
    return part.drop(columns=["_stratum"]).reset_index(drop=True)


def prepare() -> dict:
    """Download, Filter, Split, drei CSVs plus metadata.json. Rückgabe ohne Texte."""
    english, english_meta = _load_phrasebank()
    german, german_meta = _load_kenpache_german()
    english, english_stats = _clean_source(english)
    german, german_stats = _clean_source(german)
    combined = pd.concat([english, german], ignore_index=True)
    before_cross = len(combined)
    combined = combined.drop_duplicates(subset=[TEXT_COLUMN], keep="first").reset_index(drop=True)
    train, val, test = stratified_train_val_test(combined)
    _assert_partition(combined, train, val, test)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PREPARED_TRAIN_PATH, index=False)
    val.to_csv(PREPARED_VAL_PATH, index=False)
    test.to_csv(PREPARED_TEST_PATH, index=False)
    meta = {
        "seed": RANDOM_SEED,
        "test_size": PREPARE_TEST_SIZE,
        "val_size": PREPARE_VAL_SIZE,
        "train_size": 1.0 - PREPARE_TEST_SIZE - PREPARE_VAL_SIZE,
        "stratify": "label|language",
        "min_chars": MIN_TEXT_CHARS,
        "split_steps": [
            "Zuerst Test als 15 % stratifiziert nach label|language, Seed 42.",
            "Aus dem Rest Val mit Anteil 0.15/0.85, ebenfalls stratifiziert, Seed 42.",
            "Train ist der Rest, etwa 70 %.",
        ],
        "cross_source_duplicates_dropped": int(before_cross - len(combined)),
        "sources": [
            {**english_meta, **english_stats},
            {**german_meta, **german_stats},
        ],
        "rows_combined": int(len(combined)),
        "splits": {
            "train": _split_summary(train),
            "val": _split_summary(val),
            "test": _split_summary(test),
        },
    }
    PREPARED_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def _load_phrasebank() -> tuple[pd.DataFrame, dict]:
    path = _download(DATASET_ID_EN, PHRASEBANK_ZIP)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(PHRASEBANK_MEMBER)
    rows = []
    for line in raw.decode("iso-8859-1").splitlines():
        line = line.strip()
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append(
            {
                TEXT_COLUMN: " ".join(sentence.split()),
                LABEL_COLUMN: label.strip().lower(),
                LANGUAGE_COLUMN: "en",
                "source": "financial_phrasebank",
                "dataset": f"{DATASET_ID_EN}:{DATASET_CONFIG_EN}",
            }
        )
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return frame, {
        "id": DATASET_ID_EN,
        "config": DATASET_CONFIG_EN,
        "language": "en",
        "license": "CC BY-NC-SA 3.0",
        "url": "https://huggingface.co/datasets/takala/financial_phrasebank",
        "snapshot": _snapshot_id(path),
        "rows_parsed": int(len(frame)),
    }


def _load_kenpache_german() -> tuple[pd.DataFrame, dict]:
    path = _download(DATASET_ID_DE, KENPACHE_CSV)
    raw = pd.read_csv(path, encoding="utf-8")
    german = raw.loc[raw["language"].astype(str).str.strip().str.lower().eq("de")].copy()
    frame = pd.DataFrame(
        {
            TEXT_COLUMN: german["sentence"].map(lambda value: " ".join(str(value).split())),
            LABEL_COLUMN: german["label"].astype(str).str.strip().str.lower(),
            LANGUAGE_COLUMN: "de",
            "source": german["source"].astype(str).str.strip(),
            "dataset": DATASET_ID_DE,
        }
    )
    return frame.reset_index(drop=True), {
        "id": DATASET_ID_DE,
        "config": "language=de",
        "language": "de",
        "license": "academic-non-commercial (card also says Apache-2.0; stricter reading applies)",
        "url": "https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment",
        "snapshot": _snapshot_id(path),
        "rows_parsed": int(len(frame)),
    }


def _clean_source(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows_in = int(len(frame))
    work = frame.copy()
    work[TEXT_COLUMN] = work[TEXT_COLUMN].astype(str).str.strip()
    work[LABEL_COLUMN] = work[LABEL_COLUMN].astype(str).str.strip().str.lower()
    work[LANGUAGE_COLUMN] = work[LANGUAGE_COLUMN].astype(str).str.strip().str.lower()
    known = work[LABEL_COLUMN].isin(LABELS_3) & work[LANGUAGE_COLUMN].isin(ALLOWED_LANGUAGES)
    long_enough = work[TEXT_COLUMN].str.len() >= MIN_TEXT_CHARS
    work = work.loc[known & long_enough]
    after_filter = int(len(work))
    work = work.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
    stats = {
        "rows_in": rows_in,
        "rows_after_filter": after_filter,
        "dropped_short_or_invalid": int(rows_in - after_filter),
        "dropped_duplicate_text": int(after_filter - len(work)),
        "rows_kept": int(len(work)),
    }
    return work.reset_index(drop=True), stats


def _download(repo_id: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub fehlt. Bitte `pip install -r requirements.txt` ausführen."
        ) from exc
    return hf_hub_download(repo_id, filename, repo_type="dataset")


def _snapshot_id(path: str) -> str:
    parts = Path(path).parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return "unknown"


def _split_summary(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby([LANGUAGE_COLUMN, LABEL_COLUMN]).size()
    by_stratum = {f"{language}|{label}": int(count) for (language, label), count in grouped.items()}
    return {"rows": int(len(frame)), "by_language_label": by_stratum}


def _assert_partition(
    combined: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if len(train) + len(val) + len(test) != len(combined):
        raise RuntimeError("Split verliert oder verdoppelt Zeilen.")
    texts = [set(part[TEXT_COLUMN]) for part in (train, val, test)]
    if texts[0] & texts[1] or texts[0] & texts[2] or texts[1] & texts[2]:
        raise RuntimeError("Derselbe Text liegt in mehr als einem Split.")


def main() -> None:
    meta = prepare()
    print(f"Geschrieben: {PREPARED_TRAIN_PATH.parent}")
    for name, summary in meta["splits"].items():
        print(f"- {name}: {summary['rows']} Zeilen {summary['by_language_label']}")
    print(f"Metadaten: {PREPARED_META_PATH}")


if __name__ == "__main__":
    main()
