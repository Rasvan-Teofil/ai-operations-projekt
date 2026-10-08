"""Smoke-Konfiguration, ohne Basismodell und ohne Netz."""

import sys

from src.config import FINETUNED_DIR, FINETUNED_SMOKE_DIR
from src.finetune import config_from_args, finetune


def test_smoke_config_is_a_single_cpu_step():
    config = config_from_args(["--smoke"])
    assert config.smoke is True
    assert config.epochs == 1
    assert config.max_steps == 1
    assert config.max_samples == 4
    assert config.batch_size == 2
    assert config.max_length == 32
    assert config.output_dir == FINETUNED_SMOKE_DIR


def test_full_config_keeps_the_requested_budget():
    config = config_from_args(["--epochs", "2", "--lr", "0.001", "--batch-size", "4", "--max-length", "128"])
    assert config.smoke is False
    assert config.epochs == 2
    assert config.learning_rate == 0.001
    assert config.batch_size == 4
    assert config.max_length == 128
    assert config.max_steps == -1
    assert config.max_samples is None


def test_max_samples_caps_the_full_run():
    config = config_from_args(["--max-samples", "32", "--epochs", "1", "--max-length", "128"])
    assert config.smoke is False
    assert config.max_samples == 32
    assert config.epochs == 1
    assert config.max_length == 128
    assert config.output_dir == FINETUNED_DIR


def test_smoke_ignores_the_sample_cap():
    config = config_from_args(["--smoke", "--max-samples", "100"])
    assert config.max_samples == 4
    assert config.output_dir == FINETUNED_SMOKE_DIR


def test_smoke_finetune_does_not_load_transformers(monkeypatch):
    before = set(sys.modules)

    def fake_run(config, texts, labels, test_texts, test_labels):
        assert config.max_steps == 1
        assert len(texts) <= 4
        assert len(labels) == len(texts)
        return {
            "metrics": {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_negative": 0.0,
                "f1_neutral": 0.0,
                "f1_positive": 0.0,
                "inference_seconds": 0.0,
                "inference_ms_per_text": 0.0,
            },
            "confusion": None,
        }

    monkeypatch.setattr("src.finetune._run_trainer", fake_run)
    finetune(config_from_args(["--smoke"]))
    imported = set(sys.modules) - before
    assert not any(name == "transformers" or name.startswith("transformers.") for name in imported)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
