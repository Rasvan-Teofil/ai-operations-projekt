"""Render the data-scaling study from saved predictions only; no model calls."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import LABELS
from src.news_scaling import LOCAL, OUT, SPLIT_SEED, verified, read_json, write_json
from src.tracking import tracking

ORDER = ['frozen_news_v3', 'frozen_news_v4', 'minilm_fp32_news_v3', 'minilm_fp32_news_v4',
         'minilm_bf16_news_v4', 'linearsvc_news_v3', 'linearsvc_news_v4', 'qwen3_4b']
LABEL_NAMES = ['Frozen · original 244', 'Frozen · expanded 692', 'Fine-tune · original 244',
    'Fine-tune · expanded 692 FP32', 'Same expanded fit · BF16', 'LinearSVC · original 244',
    'LinearSVC · expanded 692', 'Qwen · unchanged zero-shot']


def fast_f1(y, pred):
    matrix = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
    denom = matrix.sum(0) + matrix.sum(1)
    return float(np.divide(2 * matrix.diagonal(), denom, out=np.zeros(3), where=denom != 0).mean())


def save_figure(fig, name):
    fig.savefig(OUT / (name + '.svg')); fig.savefig(OUT / (name + '.png'), dpi=160)
    plt.close(fig)
    svg = OUT / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def learning_curve():
    cv = read_json(OUT / 'cv.json'); runs = pd.DataFrame(cv['runs'])
    summary = []
    for name in ['frozen_news_v4', 'minilm_fp32_news_v4']:
        for step in [1, 2, 3]:
            rows = runs[(runs.model == name) & (runs['size'] == step)]
            assert len(rows) == 3
            summary.append({'model': name, 'size': step, 'folds': 3,
                'mean_train_rows': float(rows.fit_rows.mean()),
                **{key + '_' + stat: float(rows[key].mean() if stat == 'mean' else rows[key].std(ddof=1))
                   for key in ['balanced_language_macro_f1', 'recall_positive'] for stat in ['mean', 'sd']}})
    write_json(OUT / 'learning_curve.json', {'train_only': True,
        'interpretation': 'Three fixed group folds inside Train; nested data sizes, same development rows per fold. Error bars are sample SD across folds, not confidence intervals. Fixed final recipe not selected using these scores.',
        'points': summary})
    pd.DataFrame(summary).to_csv(OUT / 'learning_curve.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, colour, label in [('frozen_news_v4', '#64748b', 'Frozen GPU baseline'),
                                ('minilm_fp32_news_v4', '#2563eb', 'Own GPU fine-tune')]:
        rows = [r for r in summary if r['model'] == name]
        for ax, metric, title in zip(axes, ['balanced_language_macro_f1', 'recall_positive'],
                ['Mean DE/EN macro-F1 agreement', 'Positive recall']):
            ax.errorbar([r['mean_train_rows'] for r in rows], [r[metric + '_mean'] for r in rows],
                yerr=[r[metric + '_sd'] for r in rows], marker='o', capsize=5, color=colour, label=label)
            ax.set_title(title, fontsize=11); ax.set_xlabel('Mean fit rows per fold · nested subsets')
            ax.set_ylim(0, 1); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Data-size learning curve · 18 new Train-only CV runs', fontweight='bold')
    fig.text(.02, .02, 'AI-labelled · Codex-reviewed | 3 paired group folds | bars: fold SD | fresh Test and original Val unused', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92)); save_figure(fig, 'learning_curve')
    return summary


def report():
    manifest = verified(); summary = learning_curve()
    comparison = read_json(OUT / 'comparison.json'); test = pd.read_csv(LOCAL / 'test.csv')
    records = [next(r for r in comparison['results'] if r['model'] == name) for name in ORDER]
    y = np.array([LABELS.index(x) for x in test.label]); languages = test.language.to_numpy()
    predictions = {}
    for r in records:
        saved = [json.loads(x) for x in (LOCAL / (r['model'] + '_test_predictions.jsonl')).read_text().splitlines()]
        assert [x['id'] for x in saved] == test.id.tolist()
        predictions[r['model']] = np.array([np.argmax(x['scores']) for x in saved])
    unique = test['group'].unique(); rng = np.random.default_rng(SPLIT_SEED)
    indices = {g: np.flatnonzero(test['group'].to_numpy() == g) for g in unique}
    boot = {r['model']: [] for r in records}
    for _ in range(1000):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        parts = [ix[languages[ix] == lang] for lang in ['de', 'en']]
        if any(len(part) == 0 for part in parts): continue
        for name, pred in predictions.items():
            boot[name].append(np.mean([fast_f1(y[part], pred[part]) for part in parts]))
    tracking()
    for r in records:
        low, high = map(float, np.percentile(boot[r['model']], [2.5, 97.5]))
        r.update(balanced_language_macro_f1_ci_low=low, balanced_language_macro_f1_ci_high=high)
        with mlflow.start_run(run_id=r['run_id']):
            mlflow.log_metrics({'balanced_language_macro_f1_ci_low': low, 'balanced_language_macro_f1_ci_high': high})
            mlflow.set_tag('uncertainty', '1000 paired source-group bootstrap resamples of saved predictions; sampling variability only, excludes AI label error')
    changes = []
    for old, new in [('frozen_news_v3', 'frozen_news_v4'), ('minilm_fp32_news_v3', 'minilm_fp32_news_v4'),
                     ('linearsvc_news_v3', 'linearsvc_news_v4')]:
        a = next(r for r in records if r['model'] == old); b = next(r for r in records if r['model'] == new)
        delta = np.array(boot[new]) - np.array(boot[old])
        changes.append({'original': old, 'expanded': new,
            'balanced_language_macro_f1_delta': b['balanced_language_macro_f1'] - a['balanced_language_macro_f1'],
            'recall_positive_delta': b['recall_positive'] - a['recall_positive'],
            'paired_group_bootstrap_95_delta': list(map(float, np.percentile(delta, [2.5, 97.5])))})
    comparison.update(results=records, paired_data_changes=changes,
        uncertainty='1000 paired source-group bootstrap resamples, seed 20261108; percentile95 intervals cover sampling only, not AI annotation error. No model inference repeated.',
        train_rows=manifest['splits']['train']['rows'], original_train_rows=244,
        learning_curve='Three-fold nested Train-only diagnostic; full final recipe fixed in advance')
    write_json(OUT / 'comparison.json', comparison)
    columns = ['model', 'accuracy', 'macro_f1', 'balanced_language_macro_f1', 'macro_f1_de', 'macro_f1_en',
        'f1_negative', 'f1_neutral', 'f1_positive', 'recall_positive', 'recall_negative',
        'inference_ms_per_text', 'model_size_bytes', 'run_id']
    pd.DataFrame(records)[columns].to_csv(OUT / 'comparison.csv', index=False)
    colours = ['#94a3b8', '#475569', '#93c5fd', '#2563eb', '#0d9488', '#d8b4fe', '#9333ea', '#d97706']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1.55, 1, 1]})
    for ax, key, title, log in zip(axes,
        ['balanced_language_macro_f1', 'inference_ms_per_text', 'model_size_bytes'],
        ['Mean DE/EN macro-F1 agreement', 'Warm ms / text', 'Complete artifact MiB'], [False, True, True]):
        vals = [r[key] / (1048576 if key == 'model_size_bytes' else 1) for r in records]
        ax.barh(np.arange(len(records)), vals, color=colours); ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_yticks(np.arange(len(records)), LABEL_NAMES if ax == axes[0] else [''] * len(records), fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        if log: ax.set_xscale('log'); ax.set_xlim(max(min(vals) / 2, .0001), max(vals) * 8)
        else: ax.set_xlim(0, 1)
        for i, v in enumerate(vals):
            ax.text(v * 1.06 if log else v + .015, i, f'{v:.3f}' if key != 'model_size_bytes' else f'{v:.1f}', va='center', fontsize=9)
    fig.suptitle('More training coverage · original and expanded models on ONE fresh Test', fontsize=14, fontweight='bold')
    fig.text(.02, .02, f'{len(test)} native DE/EN short news texts | AI-labelled · Codex-reviewed | quantity AND sampling enrichment changed', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92)); save_figure(fig, 'comparison')
    lines = ['# Expanded news data — AI-labelled · Codex-reviewed', '',
        'The new collection adds 448 accepted training examples to the original 244:692 Train, including 206 positives instead of 45. A fresh 227-item random Test is shared by all eight conditions. The original 49 Val and49 Test remain outside fitting; the old Test is not evaluated again.', '',
        '| Condition | Mean DE/EN F1 | DE F1 | EN F1 | Positive F1 | Positive recall | ms/text | MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["name"]} | {r["balanced_language_macro_f1"]:.4f} | {r["macro_f1_de"]:.4f} | {r["macro_f1_en"]:.4f} | {r["f1_positive"]:.4f} | {r["recall_positive"]:.4f} | {r["inference_ms_per_text"]:.3f} | {r["model_size_bytes"]/1048576:.2f} |')
    lines += ['', '## Paired data changes on the same fresh Test', '']
    for r in changes:
        low, high = r['paired_group_bootstrap_95_delta']
        lines.append(f'- {r["expanded"]}: mean-language F1 change {r["balanced_language_macro_f1_delta"]:+.4f}, exploratory paired 95% interval [{low:+.4f}, {high:+.4f}]; positive recall change {r["recall_positive_delta"]:+.4f}.')
    lines += ['', 'These measure quantity plus changed class/topic coverage, not a causal effect of quantity alone. All agreements use one assistant’s reference labels, not a human annotator panel. The larger Test has 111 DE and 116 EN items, with 26 DE and 19 EN positives. Label/source exclusions preceded fitting. Inputs are historical title+lead excerpts; long-article, sentence-highlight and real newsroom performance remain unmeasured. Unknown pretraining/event overlap and annotation errors remain possible.', '',
        f'Descriptive leader under the frozen mean-language F1 /0.005 latency-size rule: **{comparison["descriptive_leader"]}**. This AI-only study does not automatically change the Qwen serving default established by the earlier human-labelled English comparison. New expanded models are experimental selectable choices. The BF16 artifact is a storage/serving derivation of the same full-Train fit, not another independent training fit.', '',
        '## Train-only learning curve', '',
        '| Model | Mean fit rows | Mean DE/EN F1 ± fold SD | Positive recall ± fold SD |',
        '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["mean_train_rows"]:.1f} | {r["balanced_language_macro_f1_mean"]:.4f} ± {r["balanced_language_macro_f1_sd"]:.4f} | {r["recall_positive_mean"]:.4f} ± {r["recall_positive_sd"]:.4f} |')
    lines += ['', 'Three fixed StratifiedGroupKFold folds inside expanded Train. Each fold has nested 1/3, 2/3, full fit-group subsets and the same development set for all sizes. Eighteen real MLflow runs compare the clean frozen GPU baseline and own GPU fine-tune. SD is variation across three folds, not a confidence interval. Final three-epoch recipe was frozen before CV; no candidate search or Val/Test selection. Larger subsets also mean more optimizer steps at a fixed epoch count. This curve uses development sets; its values are not joined to historic Test scores.', '',
        'All 26 new runs are in `article-sentiment-second-path`, phase=`wikinews-ai-v4-expanded`, stage=`cv` (18) or `final-test` (8). Required parameters, accuracy, macro/per-class F1, latency, full artifact size, and confusion matrices are logged. Old controls receive new evaluation runs; historical run metrics remain unchanged.', '',
        '[Frozen protocol](protocol.json) · [New annotation ledger](annotations.csv) · [Source credits](source_manifest.json) · [Split and input hashes](split_manifest.json) · [CV membership](cv_membership.json) · [CV run IDs](cv.json) · [Final metrics, intervals and run IDs](comparison.json)', '',
        '![Same fresh Test](comparison.png)', '', '![Train-only curve](learning_curve.png)', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))
    for r in records:
        with mlflow.start_run(run_id=r['run_id']):
            for filename in ['comparison.svg', 'learning_curve.svg', 'comparison.md']:
                mlflow.log_artifact(str(OUT / filename), 'comparison')
    print('Learning curve, paired intervals and comparison rendered from saved evidence only.', flush=True)


if __name__ == '__main__': report()
