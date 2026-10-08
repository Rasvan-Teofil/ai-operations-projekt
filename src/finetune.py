"""Fine-Tuning eines mehrsprachigen Transformers auf dem Trainings-Split.

Smoke-Check auf der CPU, ein Schritt, wenige Sätze (lädt das Basismodell
trotzdem einmal herunter, deshalb nicht in der CI):

    python -m src.finetune --smoke

Echtes Training auf einer GPU, zum Beispiel Google Colab
(Laufzeit → GPU ändern), im Projektroot:

    pip install -r requirements.txt -r requirements-transformer.txt
    python -m src.finetune --epochs 3 --lr 2e-5 --batch-size 16 --max-length 256

Gewichte liegen danach in ``models/finetuned`` (nicht im Git).
``--smoke`` schreibt nach ``models/finetuned-smoke`` und überschreibt
das echte Artefakt nicht. Die API liest nur ``models/finetuned``.

Basismodell ist ``distilbert-base-multilingual-cased``. Alternativ
``--base-model tabularisai/multilingual-sentiment-analysis``; der
Klassifikationskopf wird auf drei Labels neu aufgesetzt.
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DEFAULT_FINETUNE_BASE,
    FINETUNED_DIR,
    FINETUNED_SMOKE_DIR,
    LABELS_3,
    RANDOM_SEED,
    TEST_SIZE,
    TEXT_COLUMN,
)
from src.data import load_dataset, split_dataset
from src.evaluate import classification_metrics, confusion_frame
from src.tracking import setup_mlflow, shared_dataset_params


@dataclass(frozen=True)
class FinetuneConfig:
    base_model: str
    epochs: int
    learning_rate: float
    batch_size: int
    max_length: int
    max_steps: int
    max_samples: int | None
    smoke: bool
    output_dir: Path


def config_from_args(argv: list[str] | None = None) -> FinetuneConfig:
    """Baut die Laufkonfiguration. ``--smoke`` setzt das Budget fest."""
    parser = argparse.ArgumentParser(description="Fine-Tuning auf dem Trainings-Split.")
    parser.add_argument("--smoke", action="store_true", help="Ein Schritt, vier Sätze, CPU, kurzer Kontext.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--base-model", default=DEFAULT_FINETUNE_BASE)
    args = parser.parse_args(argv)
    if args.smoke:
        return FinetuneConfig(
            base_model=args.base_model,
            epochs=1,
            learning_rate=args.lr,
            batch_size=2,
            max_length=32,
            max_steps=1,
            max_samples=4,
            smoke=True,
            output_dir=FINETUNED_SMOKE_DIR,
        )
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8 or args.lr <= 0:
        raise ValueError("epochs, batch-size und max-length müssen positiv sein, max-length mindestens 8.")
    return FinetuneConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_steps=-1,
        max_samples=None,
        smoke=False,
        output_dir=FINETUNED_DIR,
    )


def limit_samples(texts: list[str], labels: list[str], max_samples: int | None) -> tuple[list[str], list[str]]:
    if max_samples is None:
        return list(texts), list(labels)
    return list(texts)[:max_samples], list(labels)[:max_samples]


def finetune(config: FinetuneConfig | None = None) -> dict:
    """Trainiert und schreibt Metriken nach MLflow. Der schwere Teil steckt in ``_run_trainer``."""
    config = config or config_from_args([])
    frame = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(frame)
    train_texts, train_labels = limit_samples(
        x_train[TEXT_COLUMN].astype(str).tolist(),
        y_train.astype(str).tolist(),
        config.max_samples,
    )
    test_cap = config.max_samples if config.smoke else None
    test_texts, test_labels = limit_samples(
        x_test[TEXT_COLUMN].astype(str).tolist(),
        y_test.astype(str).tolist(),
        test_cap,
    )
    result = _run_trainer(config, train_texts, train_labels, test_texts, test_labels)
    _log_finetune(config, result, n_train=len(train_texts), n_test=len(test_texts))
    print(
        f"Fine-Tuning ({'smoke' if config.smoke else 'voll'}) → {config.output_dir} "
        f"macro-F1={result['metrics']['f1_macro']:.3f}"
    )
    return result


def _log_finetune(config: FinetuneConfig, result: dict, n_train: int, n_test: int) -> None:
    import mlflow

    setup_mlflow()
    shared = shared_dataset_params(n_train, n_test, RANDOM_SEED, TEST_SIZE)
    with mlflow.start_run(run_name="finetuned-smoke" if config.smoke else "finetuned"):
        mlflow.log_params(
            {
                **shared,
                "base_model": config.base_model,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "max_steps": config.max_steps,
                "smoke": str(config.smoke).lower(),
            }
        )
        mlflow.log_metrics(result["metrics"])
        mlflow.set_tag("role", "finetuned")
        mlflow.set_tag("smoke", "true" if config.smoke else "false")
        confusion = result.get("confusion")
        if confusion is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "confusion_matrix.csv"
                confusion.to_csv(path)
                mlflow.log_artifact(str(path))
        meta = config.output_dir / "training_meta.json"
        if meta.exists():
            mlflow.log_artifact(str(meta))


def _run_trainer(
    config: FinetuneConfig,
    texts: list[str],
    labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
) -> dict:
    """Lädt das Basismodell, trainiert einen 3-Klassen-Kopf, bewertet den Testsplit."""
    if config.smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch oder Transformers fehlen. "
            "Bitte `pip install -r requirements-transformer.txt` ausführen."
        ) from exc

    from src.hf_models import HuggingFaceSentimentModel

    set_seed(RANDOM_SEED)
    label2id = {name: index for index, name in enumerate(LABELS_3)}
    id2label = {index: name for name, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(LABELS_3),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    train_dataset = _encode_dataset(torch, tokenizer, texts, labels, label2id, config.max_length)
    argument_kwargs = {
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": RANDOM_SEED,
        "disable_tqdm": True,
    }
    if config.max_steps > 0:
        argument_kwargs["max_steps"] = config.max_steps
    if config.smoke:
        argument_kwargs["use_cpu"] = True

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        argument_kwargs["output_dir"] = tmp
        training_args = _training_arguments(TrainingArguments, argument_kwargs, config.smoke)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    meta = {
        "base_model": config.base_model,
        "labels": list(LABELS_3),
        "smoke": config.smoke,
        "max_length": config.max_length,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
    }
    (config.output_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    loaded = HuggingFaceSentimentModel(
        name="finetuned",
        model_version=f"finetuned:{config.base_model}",
        model_id=str(config.output_dir),
        max_length=min(512, config.max_length),
        role="finetuned",
        notes="Fine-Tuning",
    ).load()
    import time

    started = time.perf_counter()
    predicted = loaded.predict(test_texts) if test_texts else []
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(test_labels, predicted) if test_texts else {
        "accuracy": 0.0,
        "f1_macro": 0.0,
        "f1_negative": 0.0,
        "f1_neutral": 0.0,
        "f1_positive": 0.0,
    }
    metrics["inference_seconds"] = float(elapsed)
    metrics["inference_ms_per_text"] = float(elapsed / len(test_texts) * 1000.0) if test_texts else 0.0
    confusion = confusion_frame(test_labels, predicted) if test_texts else None
    return {"metrics": metrics, "confusion": confusion}


def _encode_dataset(torch, tokenizer, texts, labels, label2id, max_length):
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    class EncodedTextDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            item = {key: value[index] for key, value in encoded.items()}
            item["labels"] = int(label2id[labels[index]])
            return item

    return EncodedTextDataset()


def _training_arguments(training_arguments_cls, kwargs: dict, smoke: bool):
    try:
        return training_arguments_cls(**kwargs)
    except TypeError:
        if not smoke or "use_cpu" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback.pop("use_cpu", None)
        fallback["no_cuda"] = True
        return training_arguments_cls(**fallback)


def main(argv: list[str] | None = None) -> None:
    config = config_from_args(argv)
    if config.smoke:
        print("Smoke-Fine-Tuning: 1 Schritt, 4 Trainingssätze, CPU. Kein belastbares Modell.")
    finetune(config)


if __name__ == "__main__":
    main()
