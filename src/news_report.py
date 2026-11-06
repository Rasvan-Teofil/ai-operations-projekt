"""Render saved pilot predictions; never reload or re-evaluate a model."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow

from src.config import LABELS
from src.news_data import LOCAL, OUT, SEED
from src.news_experiment import scores, verified, read_json, write_json
from src.tracking import tracking


def report():
    verified()
    comparison = read_json(OUT / 'comparison.json')
    test = pd.read_csv(LOCAL / 'test.csv')
    y = np.array([LABELS.index(x) for x in test.label])
    languages = test.language.to_numpy()
    records = comparison['results']
    predictions = {}
    for r in records:
        saved = [json.loads(x) for x in (LOCAL / f'{r["model"]}_test_predictions.jsonl').read_text().splitlines()]
        assert [x['id'] for x in saved] == test.id.tolist()
        predictions[r['model']] = np.array([np.argmax(x['scores']) for x in saved])
    rng = np.random.default_rng(SEED)
    unique = test['group'].unique()
    indices = {g: np.flatnonzero(test['group'].to_numpy() == g) for g in unique}
    boot = {r['model']: [] for r in records}
    for _ in range(1000):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        if not all(np.any(languages[ix] == lang) for lang in ['de', 'en']):
            continue
        for r in records:
            values = []
            for lang in ['de', 'en']:
                part = ix[languages[ix] == lang]
                values.append(scores(y[part], predictions[r['model']][part])['macro_f1'])
            boot[r['model']].append(np.mean(values))
    tracking()
    for r in records:
        low, high = np.percentile(boot[r['model']], [2.5, 97.5])
        r.update(balanced_language_macro_f1_ci_low=float(low), balanced_language_macro_f1_ci_high=float(high))
        with mlflow.start_run(run_id=r['run_id']):
            mlflow.log_metrics({'balanced_language_macro_f1_ci_low': float(low), 'balanced_language_macro_f1_ci_high': float(high)})
            mlflow.set_tag('uncertainty', 'Exploratory paired group bootstrap of saved predictions, 1000 samples; does not include annotation error')
    comparison['uncertainty'] = 'Exploratory paired group bootstrap, 1000 resamples, seed20261107, 95% percentile intervals; no model rerun. AI annotation error is not covered.'
    comparison['paired_intervals'] = {}
    for name in ['frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3']:
        delta = np.array(boot['qwen3_4b']) - np.array(boot[name])
        comparison['paired_intervals']['qwen_minus_' + name] = list(map(float, np.percentile(delta, [2.5, 97.5])))
    write_json(OUT / 'comparison.json', comparison)
    columns = ['model', 'accuracy', 'macro_f1', 'balanced_language_macro_f1', 'macro_f1_de', 'macro_f1_en',
        'f1_negative', 'f1_neutral', 'f1_positive', 'inference_ms_per_text', 'model_size_bytes', 'run_id']
    pd.DataFrame(records)[columns].to_csv(OUT / 'comparison.csv', index=False)
    order = ['frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3', 'qwen3_4b']
    rows = [next(r for r in records if r['model'] == name) for name in order]
    labels = ['Frozen GPU baseline', 'New fine-tune · FP32', 'Same fine-tune · BF16', 'New LinearSVC', 'Qwen · zero-shot']
    colours = ['#64748b', '#2563eb', '#0d9488', '#a855f7', '#d97706']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), gridspec_kw={'width_ratios': [1.55, 1, 1]})
    for ax, key, title, log in zip(axes,
            ['balanced_language_macro_f1', 'inference_ms_per_text', 'model_size_bytes'],
            ['Mean DE/EN macro-F1 agreement', 'Warm ms / text', 'Complete artifact MiB'], [False, True, True]):
        vals = [r[key] / (1048576 if key == 'model_size_bytes' else 1) for r in rows]
        ax.barh(np.arange(len(rows)), vals, color=colours)
        ax.invert_yaxis(); ax.set_title(title, fontsize=10, pad=12)
        ax.set_yticks(np.arange(len(rows)), labels if ax == axes[0] else [''] * len(rows), fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        if log: ax.set_xscale('log'); ax.set_xlim(max(min(vals) / 2, .0001), max(vals) * 8)
        else: ax.set_xlim(0, 1)
        for i, v in enumerate(vals): ax.text(v * 1.06 if log else v + .015, i, f'{v:.3f}' if key != 'model_size_bytes' else f'{v:.1f}', va='center', fontsize=9)
    fig.suptitle('New Wikinews pilot · AI-labelled · Codex-reviewed', fontsize=15, fontweight='bold', x=.53)
    fig.text(.02, .02, f'{len(test)} held-out short news texts · native DE + EN · single AI annotation source · no human-gold accuracy claim', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(OUT / 'comparison.svg'); fig.savefig(OUT / 'comparison.png', dpi=160); plt.close(fig)
    svg = OUT / 'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    lines = ['# New Wikinews pilot — AI-labelled · Codex-reviewed', '',
        'This is a separate feasibility experiment, not a replacement for the historical human-labelled English benchmark. All figures measure agreement with one assistant’s reference annotations.', '',
        '| Model | Mean DE/EN F1 | DE F1 | EN F1 | Positive F1 | ms/text | MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["balanced_language_macro_f1"]:.4f} | {r["macro_f1_de"]:.4f} | {r["macro_f1_en"]:.4f} | {r["f1_positive"]:.4f} | {r["inference_ms_per_text"]:.3f} | {r["model_size_bytes"]/1048576:.2f} |')
    lines += ['', 'Qwen remains the default: it leads this pilot and the earlier independent human-labelled English comparison. The new fine-tune did not outperform the frozen baseline here. All four fitted classifiers have zero positive-class F1 on the ten positive test examples; Qwen’s positive F1 is also weak. These results do not establish an improvement over the old versions.', '',
        'The FP32 and BF16 conditions use the same newly trained weights. BF16 reduces size but is slightly slower in this single short measurement. It is not a second independent fit.', '',
        '244 Train / 49 Val / 49 Test; Val unused. The test has 24 DE and 25 EN items, with only 4 DE and 6 EN positives. All candidates use the exact same test texts and AI labels. There are no machine translations. Short inputs, small sample size, one AI annotator, residual event overlap and unknown pretraining contamination limit conclusions. Do not compare these scores numerically with 0.720 or earlier benchmark figures.', '',
        'Exploratory paired group bootstrap of saved predictions: 1,000 resamples, seed20261107, 95% percentile intervals in comparison.json. These intervals cover sampling variability, not AI annotation mistakes. No extra model evaluation was performed.', '',
        '[Frozen protocol](protocol.json) · [AI labels and short rationales](annotations.csv) · [split and source hashes](split_manifest.json) · [source credits](source_manifest.json) · [CSV](comparison.csv) · [run IDs and confusion matrices](comparison.json)', '',
        '![Measured pilot](comparison.png)', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))
    for r in records:
        with mlflow.start_run(run_id=r['run_id']):
            mlflow.log_artifact(str(OUT / 'comparison.svg'), 'comparison')
            mlflow.log_artifact(str(OUT / 'comparison.md'), 'comparison')
    print('Rendered saved results and exploratory intervals; no model inference repeated.', flush=True)


if __name__ == '__main__': report()
