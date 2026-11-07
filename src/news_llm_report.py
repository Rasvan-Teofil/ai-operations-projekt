"""Matched-task reports from saved CV/Test predictions only; no model inference."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from src.config import LABELS
from src.news_llm_experiment import LOCAL, OUT, SEED, QWEN, NAMES, verified, read_json, write_json
from src.news_scaling_report import fast_f1
from src.tracking import tracking

ORDER = ['frozen_news_v4', 'minilm_bf16_news_v4', 'linearsvc_news_v4', *QWEN]
SHORT = ['Frozen MiniLM', 'MiniLM BF16 fine-tune', 'LinearSVC', 'Qwen · financial prompt',
         'Qwen · matched news prompt', 'Qwen · six examples', 'Qwen · own LoRA']
COLOURS = ['#64748b', '#2563eb', '#9333ea', '#fbbf24', '#d97706', '#0d9488', '#be123c']


def figure(fig, name):
    fig.savefig(OUT / (name + '.svg')); fig.savefig(OUT / (name + '.png'), dpi=160); plt.close(fig)
    path = OUT / (name + '.svg')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def development():
    cv = read_json(OUT / 'cv.json'); rows = cv['runs']; summary = []
    for name in QWEN:
        runs = [r for r in rows if r['model'] == name]; assert len(runs) == 3
        summary.append({'model': name, 'name': NAMES[name], 'folds': 3,
            **{metric + '_' + stat: float(np.mean([r[metric] for r in runs]) if stat == 'mean'
                    else np.std([r[metric] for r in runs], ddof=1))
               for metric in ['balanced_language_macro_f1', 'recall_positive', 'inference_ms_per_text']
               for stat in ['mean', 'sd']}})
    references = []
    for name in ['frozen_news_v4', 'minilm_fp32_news_v4']:
        runs = [r for r in cv['reused_small_model_runs'] if r['model'] == name]; assert len(runs) == 3
        references.append({'model': name, 'name': 'MiniLM FP32 reference' if 'fp32' in name else 'Frozen MiniLM reference',
            **{metric + '_mean': float(np.mean([r[metric] for r in runs]))
               for metric in ['balanced_language_macro_f1', 'recall_positive']},
            'run_ids': [r['run_id'] for r in runs]})
    write_json(OUT / 'development.json', {'train_only': True, 'points': summary, 'historical_same_fold_references': references,
        'interpretation': 'Same three Train-only source-group dev folds; fixed prompt/example/LoRA conditions. Bars are fold SD. Small-model FP32 reference scores reused, not recomputed; final small-model deployment is BF16 from same fit. No Val or fresh-Test model choice.'})
    pd.DataFrame(summary).to_csv(OUT / 'development.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, key, title in zip(axes, ['balanced_language_macro_f1', 'recall_positive'],
                             ['Mean DE/EN macro-F1 agreement', 'Positive recall']):
        ax.bar(np.arange(4), [r[key + '_mean'] for r in summary], color=COLOURS[3:],
            yerr=[r[key + '_sd'] for r in summary], capsize=5)
        ax.set_xticks(np.arange(4), ['Financial\nprompt', 'News\nrubric', 'News + six\nexamples', 'Own\nLoRA'], fontsize=9)
        ax.set_ylim(0, 1); ax.set_title(title, fontsize=11); ax.spines[['top', 'right']].set_visible(False)
        for ref, colour in zip(references, ['#64748b', '#2563eb']):
            ax.axhline(ref[key + '_mean'], color=colour, ls='--', lw=1.3, label=ref['name'])
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('LLM task adaptation · same three Train-only group folds', fontweight='bold', fontsize=13)
    fig.text(.02, .02, 'AI-labelled · Codex-reviewed | 12 new LLM runs | SD across folds, not CI | fixed recipes; original Val and fresh Test unused', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .92)); figure(fig, 'development')
    return summary


def training_cost(store, record):
    """Additional LoRA cost or historical control fit cost; no invented zero-cost training."""
    source = record.get('original_run_id') or record['run_id']
    run = store.get_run(source)
    if 'derived_from_run' in run.data.params:
        source = run.data.params['derived_from_run']; run = store.get_run(source)
    return {'training_seconds': float(run.data.metrics.get('training_seconds', 0.)),
            'training_cost_run_id': source, 'cost_scope': 'Historical original fit' if record.get('original_run_id') else 'New adapter fit' if record['model'] == 'qwen_lora_v5' else 'No new weight training; pretraining cost excluded'}


def report():
    m = verified(); summary = development(); comparison = read_json(OUT / 'comparison.json')
    test = pd.read_csv(LOCAL / 'test.csv'); y = np.array([LABELS.index(x) for x in test.label])
    languages = test.language.to_numpy(); records = [next(r for r in comparison['results'] if r['model'] == name) for name in ORDER]
    predictions = {}
    for record in records:
        rows = [json.loads(x) for x in (LOCAL / (record['model'] + '_test_predictions.jsonl')).read_text().splitlines()]
        assert [r['id'] for r in rows] == test.id.tolist()
        predictions[record['model']] = np.array([np.argmax(r['scores']) for r in rows])
    unique = test['group'].unique(); indices = {g: np.flatnonzero(test['group'].to_numpy() == g) for g in unique}
    rng = np.random.default_rng(SEED); boot = {name: [] for name in ORDER}
    for _ in range(1000):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        parts = [ix[languages[ix] == lang] for lang in ['de', 'en']]
        if any(len(part) == 0 for part in parts): continue
        for name in ORDER: boot[name].append(np.mean([fast_f1(y[p], predictions[name][p]) for p in parts]))
    tracking(); store = MlflowClient()
    for r in records:
        low, high = map(float, np.percentile(boot[r['model']], [2.5, 97.5]))
        r.update(balanced_language_macro_f1_ci_low=low, balanced_language_macro_f1_ci_high=high, **training_cost(store, r))
        with mlflow.start_run(run_id=r['run_id']):
            mlflow.log_metrics({'balanced_language_macro_f1_ci_low': low, 'balanced_language_macro_f1_ci_high': high})
            mlflow.set_tag('uncertainty', '1000 paired source-group bootstrap resamples of saved predictions; sampling variability only, not AI label error')
    changes = []
    for old, new in [('qwen3_4b', 'qwen_news_v5'), ('qwen_news_v5', 'qwen_fewshot_v5'),
                     ('qwen_news_v5', 'qwen_lora_v5'), ('minilm_bf16_news_v4', 'qwen_lora_v5')]:
        a = next(r for r in records if r['model'] == old); b = next(r for r in records if r['model'] == new)
        changes.append({'from': old, 'to': new, 'mean_language_f1_delta': b['balanced_language_macro_f1'] - a['balanced_language_macro_f1'],
            'paired_group_bootstrap_95_delta': list(map(float, np.percentile(np.array(boot[new]) - np.array(boot[old]), [2.5, 97.5])))})
    comparison.update(results=records, paired_changes=changes, development_points=summary,
        uncertainty='Exploratory paired source-group bootstrap of saved predictions, 1000 resamples, seed 20261109. Sampling variability only; annotation error and multiple-comparison adjustment not included.',
        distinction_from_v4='A different 227-item Test, hash 723673ee...; v4 hash 6f02c3a6... is not reused. Compare methods only within this new table.',
        training_rows=m['splits']['train']['rows'])
    write_json(OUT / 'comparison.json', comparison)
    columns = ['model', 'accuracy', 'macro_f1', 'balanced_language_macro_f1', 'macro_f1_de', 'macro_f1_en',
        'f1_negative', 'f1_neutral', 'f1_positive', 'recall_positive', 'recall_negative', 'inference_ms_per_text',
        'model_size_bytes', 'model_incremental_storage_bytes', 'training_seconds', 'cost_scope', 'run_id']
    pd.DataFrame(records)[columns].to_csv(OUT / 'comparison.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1.55, 1, 1]})
    for ax, key, title, log in zip(axes,
            ['balanced_language_macro_f1', 'inference_ms_per_text', 'model_size_bytes'],
            ['Mean DE/EN macro-F1 agreement', 'Warm ms / text', 'Standalone artifact MiB'], [False, True, True]):
        vals = [r[key] / (1048576 if key == 'model_size_bytes' else 1) for r in records]
        ax.barh(np.arange(len(records)), vals, color=COLOURS); ax.invert_yaxis()
        ax.set_yticks(np.arange(len(records)), SHORT if ax == axes[0] else [''] * len(records), fontsize=9)
        ax.set_title(title, fontsize=11); ax.spines[['top', 'right']].set_visible(False)
        if log: ax.set_xscale('log'); ax.set_xlim(max(min(vals) / 2, .0001), max(vals) * 8)
        else: ax.set_xlim(0, 1)
        for i, v in enumerate(vals): ax.text(v * 1.06 if log else v + .015, i, f'{v:.3f}' if key != 'model_size_bytes' else f'{v:.1f}', va='center', fontsize=9)
    fig.suptitle('Matched news task · prompt / examples / own LoRA · ONE fresh Test', fontweight='bold', fontsize=14)
    fig.text(.02, .02, f'{len(test)} new DE/EN short news texts | AI-labelled · Codex-reviewed | same 692 Train for supervised fits; no preassigned winner', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .92)); figure(fig, 'comparison')
    leader = next(r for r in records if r['model'] == comparison['descriptive_leader'])
    few = next(r for r in records if r['model'] == 'qwen_fewshot_v5')
    lora = next(r for r in records if r['model'] == 'qwen_lora_v5')
    lines = ['# Matched-task LLM adaptation — AI-labelled · Codex-reviewed', '',
        'The previous data-expansion experiment compared a task-trained classifier with unchanged financial-domain zero-shot Qwen on broad news. It fairly measured the value of new training coverage, but did not establish a general architecture ranking. This follow-up tests the suspected task/adaptation mismatch without assigning a winner in advance.', '',
        '**Same 692 Train inputs and labels as v4; no new Train data.** Qwen conditions: original financial prompt, fixed general-news rubric, same rubric with six seeded Train-only demonstrations, and an own rank 8 LoRA adaptation. Small-model frozen baseline, BF16 fine-tune and LinearSVC remain unchanged controls. LoRA has 2,949,120 trainable parameters, a frozen BF16 base and FP32 adapters/Adam states; three full epochs, microbatch 1 / accumulation 16, LR 1e-4, no quantization or truncation. It learns conditional A/B/C next-token classification, not free-text generation. The discarded Train-only memory probe fit the longest input (533 tokens) in 8.34 GB allocated; no accuracy claim comes from it.', '',
        f'**Different fresh Test: {len(test)} inputs, exact SHA-256 `{comparison["test_sha256"]}`.** Random 120/language before labels from the same recent 1500 page pool; all 1080 historical Wikinews inputs and linked/source/near-duplicate groups excluded. 239 raw inputs, 12 annotation exclusions, 227 retained: 132 negative / 58 neutral / 37 positive; DE 109, EN 118. These are not the previous v4 Test despite the coincidentally equal 227 count. No class-hint enrichment or balancing on Test. One AI annotator created/reviewed exact-input labels before candidate results. Exact assistant serving revision is unavailable. Old Val/Test files are not used to fit or select examples, and no old Test or run metrics are rescored/overwritten.', '',
        '| Condition | Mean DE/EN F1 | DE | EN | Positive recall | ms/text | Standalone MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["name"]} | {r["balanced_language_macro_f1"]:.4f} | {r["macro_f1_de"]:.4f} | {r["macro_f1_en"]:.4f} | {100*r["recall_positive"]:.1f}% | {r["inference_ms_per_text"]:.3f} | {r["model_size_bytes"]/1048576:.2f} |')
    lines += ['', f'Descriptive leader under the frozen mean-language F1 rule (within 0.005: latency then size): **{leader["name"]}**. The result is agreement with one AI reference annotator, not human-gold accuracy or proof that an architecture is universally better. The benchmark serving default remains the historical Qwen selection under the predeclared AI-only policy; new profiles are experimental website choices.', '',
        f'LoRA minus Few-Shot is {lora["balanced_language_macro_f1"] - few["balanced_language_macro_f1"]:+.4f} F1, with {lora["inference_ms_per_text"]:.3f} versus {few["inference_ms_per_text"]:.3f} ms/text. This is a descriptive point ordering, not a claim of statistically established superiority for that pair. LoRA catches {lora["confusion_matrix"]["all"][0][0]}/132 negatives versus {few["confusion_matrix"]["all"][0][0]}/132, but only {lora["confusion_matrix"]["all"][2][2]}/37 positives versus {few["confusion_matrix"]["all"][2][2]}/37. The fixed macro-F1 criterion does not imply improvement for every class.', '',
        'Few-Shot is below the news prompt in two development folds but above it on this fresh Test. The conditions were retained before either result; this reversal is not used to revise examples, labels or training. CV means and fresh-Test scores measure different samples and must not be joined as a single numerical progress curve.', '', '## Paired changes', '']
    for r in changes:
        low, high = r['paired_group_bootstrap_95_delta']
        lines.append(f'- {r["from"]} → {r["to"]}: {r["mean_language_f1_delta"]:+.4f}, exploratory paired 95% interval [{low:+.4f}, {high:+.4f}].')
    lines += ['', 'Intervals use 1,000 source-group bootstrap resamples of saved predictions, seed 20261109; no repeated model inference. They cover sampling variability, not systematic annotation error, and are not adjusted for multiple comparisons.', '',
        '## Train-only development', '', '| Qwen condition | Mean F1 ± fold SD | Positive recall ± fold SD |', '|---|---:|---:|']
    for r in summary:
        lines.append(f'| {r["name"]} | {r["balanced_language_macro_f1_mean"]:.4f} ± {r["balanced_language_macro_f1_sd"]:.4f} | {r["recall_positive_mean"]:.4f} ± {r["recall_positive_sd"]:.4f} |')
    lines += ['', 'Three unchanged v4 full-size source-group folds within Train, four Qwen conditions/fold=12 new CV runs. Six examples are sampled independently from that fold’s fit rows (one/language/class), never from dev. The old small-model FP32 full-fold scores are reused with their original run IDs and clearly marked as references; they are not new runs or new BF16 measurements. All recipes were fixed before CV: no prompt sweep, epoch choice or hyperparameter search. The full-Train LoRA subsequently uses all 692 inputs each epoch. All seven conditions are frozen before the final fresh-Test audit, once per text/condition with exclusive completion lock and interruption checkpoints.', '',
        '## Cost and serving', '', 'Full standalone size includes the shared Qwen base plus each profile/adapter. Incremental on-disk size is separately in CSV/JSON; it is not presented as a tiny standalone LLM. Prompt/Few-Shot update no weights, but have preprocessing and inference cost. Few-shot context overhead is included in measured batch-one latency. All methods exclude pretraining cost; controls retain historical fit costs rather than pretending they were free.', '',
        '| Condition | Additional/historical fit seconds | Cost scope |', '|---|---:|---|']
    for r in records: lines.append(f'| {r["name"]} | {r["training_seconds"]:.2f} | {r["cost_scope"]} |')
    lines += ['', 'One isolated RTX 5060 Ti benchmark worker; API paused, 4 CPU threads, 20 separate synthetic warmups, synchronized batch 1 timing including tokenization/transfers but excluding loading/network. The API shares one local Qwen base across prompts and the adapter. Adapter state is serialized; unadapted requests explicitly disable the adapter. Offline unit checks and synthetic before/after regression controls verify routing/state isolation; no held-out test is reused for API testing.', '',
        '## Interpretation and limits', '', 'This gives the LLM the same task definition and access to the same training labels, including its own supervised adaptation. It does not equalize parameter counts, optimizer settings, total updates, context length or compute. Small-model Train truncation/chunk inference and LLM context policies differ and remain disclosed. A larger LLM can help but cannot be declared the winner by design. Single-AI labels, short historical source leads, unknown pretraining/event overlap and small positive-class language cells limit conclusions. Source licensing is retained; data/weights/demo texts remain outside Git.', '',
        'All 19 new scored runs use `article-sentiment-second-path`, phase=`wikinews-llm-adaptation-v5`, stage=`cv` (12) or `final-test` (7). Accuracy, macro/per-class F1, recalls, language metrics, latency, standalone size, confusion matrices, hashes and runtime/Git parameters are logged. The memory probe is engineering evidence, not an unscored model-accuracy run.', '',
        '[Frozen protocol and exact prompt](protocol.json) · [AI labels/rationales](annotations.csv) · [Source credits](source_manifest.json) · [Split](split_manifest.json) · [CV example/membership audit](cv_membership.json) · [CV run IDs](cv.json) · [Frozen artifact/dependency hashes](trained.json) · [Final metrics and intervals](comparison.json)', '',
        'Implementation sources: [official Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [PEFT LoRA documentation](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/lora), [PEFT precision/adapter guidance](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/troubleshooting). These document the method; project measurements above are from actual local runs.', '',
        '![Fresh Test comparison](comparison.png)', '', '![Train-only development](development.png)', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))
    for r in records:
        with mlflow.start_run(run_id=r['run_id']):
            for name in ['comparison.svg', 'development.svg', 'comparison.md']: mlflow.log_artifact(str(OUT / name), 'comparison')
    print('Matched-task results and development rendered from saved predictions; no model rerun.', flush=True)


if __name__ == '__main__': report()
