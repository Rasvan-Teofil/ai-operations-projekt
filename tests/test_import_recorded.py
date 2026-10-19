"""Der CPU-Import legt beide Durchläufe in denselben MLflow-Store."""

import mlflow

from src.import_recorded import SECOND_EXPERIMENT, import_recorded


def _runs(store, experiment: str):
    mlflow.set_tracking_uri(store.resolve().as_uri())
    frame = mlflow.search_runs(experiment_names=[experiment], max_results=200)
    return frame


def test_import_recorded_is_idempotent_and_tagged(tmp_path):
    store = tmp_path / "mlruns"
    first = import_recorded(store)
    second = import_recorded(store)
    assert first["logged"] > 0
    assert first["skipped"] == 0
    assert second["logged"] == 0
    assert second["skipped"] == first["logged"]

    comparison = _runs(store, "article-sentiment-comparison")
    assert not comparison.empty
    keys = set(comparison["tags.recorded_key"])
    assert "comparison/first-own-test/tfidf_linearsvc" in keys
    assert "comparison/sentfin-external/second_path_v2" in keys
    assert "comparison/sentfin-external/colleague_main_default" in keys
    assert "comparison/second-source-cv/R3_deploy" in keys

    linearsvc = comparison[comparison["tags.recorded_key"] == "comparison/first-own-test/tfidf_linearsvc"].iloc[0]
    assert linearsvc["tags.iteration"] == "1-first-path"
    assert linearsvc["tags.source"] == "recorded-report"
    assert linearsvc["metrics.f1_macro"] == 0.7202

    peer = comparison[comparison["tags.recorded_key"] == "comparison/sentfin-external/colleague_main_default"].iloc[0]
    v2 = comparison[comparison["tags.recorded_key"] == "comparison/sentfin-external/second_path_v2"].iloc[0]
    assert peer["tags.iteration"] == "1-first-path"
    assert v2["tags.iteration"] == "2-second-path"
    assert peer["tags.comparable_group"] == "sentfin-external"
    assert v2["tags.comparable_group"] == "sentfin-external"
    assert abs(v2["metrics.macro_f1_en"] - 0.7293498260817141) < 1e-12

    r3 = comparison[comparison["tags.recorded_key"] == "comparison/second-source-cv/R3_deploy"].iloc[0]
    assert abs(r3["metrics.macro_f1"] - 0.6728847544039361) < 1e-12

    native_second = _runs(store, SECOND_EXPERIMENT)
    assert "second-own-test/finetuned" in set(native_second["tags.recorded_key"])
    finetuned = native_second[native_second["tags.recorded_key"] == "second-own-test/finetuned"].iloc[0]
    assert finetuned["tags.iteration"] == "2-second-path"
    assert abs(finetuned["metrics.macro_f1"] - 0.7186645573333518) < 1e-12
