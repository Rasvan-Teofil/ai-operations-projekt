"""Complete existing clipping pairs; no further hyperparameter selection."""
from src.studies.optimizer_ablation.design import configurations as factorial_configs
from src.studies.optimizer_validation.design import configurations as validation_configs, SEEDS

PARAMETERS = ('learning_rate', 'weight_decay', 'warmup_fraction')
METRICS = ('primary_score', 'macro_f1_de', 'macro_f1_en', 'macro_f1_ja')


def configurations():
    """Seven absent controls, in the original fixed validation block order."""
    return [dict(c, id=c['id'].replace('baseline', 'baseline_no_clip'),
                 recipe='baseline_no_clip', gradient_clip=None)
            for c in validation_configs() if c['recipe'] == 'baseline'
            and not (c['fold'] == 0 and c['seed'] in SEEDS[:2])]


def reference(study, run_id):
    return {kind: f'reports/{study}/{kind}/{run_id}.json'
            for kind in ('runs', 'predictions', 'telemetry')}


def pair_id(config):
    return (f'f{config["fold"]}-s{config["seed"]}'
            + ''.join(f'-{k}={config[k]:g}' for k in PARAMETERS))


def pairs():
    """30 distinct pairs / 60 fits, with four pairs shared between views."""
    factorial = factorial_configs()
    rows = {}
    for c in factorial:
        if c['gradient_clip'] is None:
            continue
        off = next(r for r in factorial if r['gradient_clip'] is None
                   and r['seed'] == c['seed'] and all(r[k] == c[k] for k in PARAMETERS))
        config = {k: c[k] for k in (*PARAMETERS, 'seed', 'gradient_clip')}
        config['fold'] = 0
        key = pair_id(config)
        rows[key] = dict(id=key, config=config, factorial=True, cv_recipe=None,
                         with_clipping=reference('optimizer_ablation', c['id']),
                         without_clipping=reference('optimizer_ablation', off['id']))
    for c in validation_configs():
        if c['gradient_clip'] is None:
            continue
        key = pair_id(c)
        if key in rows:
            rows[key]['cv_recipe'] = c['recipe']
            continue
        off_study = 'clipping_control' if c['recipe'] == 'baseline' else 'optimizer_validation'
        rows[key] = dict(id=key, config={k: c[k] for k in (*PARAMETERS, 'fold', 'seed', 'gradient_clip')},
                         factorial=False, cv_recipe=c['recipe'],
                         with_clipping=reference('optimizer_validation', c['id']),
                         without_clipping=reference(off_study, c['id'] + '_no_clip'))
    return list(rows.values())


def validate_pair(pair, clipped, unclipped):
    for name, result, threshold in (('with', clipped, 1.0), ('without', unclipped, None)):
        c = result['config']
        if (c['gradient_clip'] != threshold or c['seed'] != pair['config']['seed']
                or c.get('fold', 0) != pair['config']['fold']
                or any(c[k] != pair['config'][k] for k in PARAMETERS)):
            raise ValueError('Non-matching clipping pair: ' + name)
    if clipped['mlflow_run_id'] == unclipped['mlflow_run_id']:
        raise ValueError('A clipping pair must contain two distinct fits')
    if ('fit_ids' in clipped and 'fit_ids' in unclipped
            and (clipped['fit_ids'] != unclipped['fit_ids'] or clipped['dev_ids'] != unclipped['dev_ids'])):
        raise ValueError('Clipping pair membership differs')


def analyze_cv(predictions, groups, bootstrap_samples=2000):
    """Pool folds before F1. The sign is always WITH minus WITHOUT clipping."""
    import numpy as np
    from src.studies.optimizer_validation.design import score
    expected = {(recipe, seed, fold, arm) for recipe in ('baseline', 'candidate')
                for seed in SEEDS for fold in range(3) for arm in ('with', 'without')}
    if set(predictions) != expected:
        return dict(status='pending', available_cv_arms=len(predictions), expected_cv_arms=36)
    pooled = {}
    for recipe in ('baseline', 'candidate'):
        for seed in SEEDS:
            for arm in ('with', 'without'):
                records = [r for fold in range(3) for r in predictions[recipe, seed, fold, arm]]
                if len(records) != len(groups) or {r['id'] for r in records} != set(groups):
                    raise ValueError('OOF coverage must be exactly all Train IDs once')
                pooled[recipe, seed, arm] = records
    metadata = None
    for records in pooled.values():
        current = {r['id']: (r['reference'], r['language'], r['text_sha256']) for r in records}
        if metadata is not None and metadata != current:
            raise ValueError('Paired prediction metadata differs')
        metadata = current
    analyses = {}
    for recipe in ('baseline', 'candidate'):
        per_seed = {}
        for seed in SEEDS:
            scores = {arm: score(pooled[recipe, seed, arm]) for arm in ('with', 'without')}
            outside = {arm: score([r for fold in (1, 2) for r in predictions[recipe, seed, fold, arm]])
                       for arm in ('with', 'without')}
            per_seed[str(seed)] = dict(oof=scores,
                with_minus_without={m: scores['with'][m] - scores['without'][m] for m in METRICS},
                excluding_selection_fold_with_minus_without={m: outside['with'][m] - outside['without'][m] for m in METRICS})
        means = {m: float(np.mean([s['with_minus_without'][m] for s in per_seed.values()])) for m in METRICS}
        intervals = cluster_intervals({(seed, arm): pooled[recipe, seed, arm]
                                      for seed in SEEDS for arm in ('with', 'without')}, groups, bootstrap_samples)
        analyses[recipe] = dict(per_seed=per_seed, mean_with_minus_without=means,
            mean_oof={arm: {m: float(np.mean([s['oof'][arm][m] for s in per_seed.values()])) for m in METRICS}
                      for arm in ('with', 'without')},
            excluding_selection_fold_mean_with_minus_without={m: float(np.mean([s['excluding_selection_fold_with_minus_without'][m] for s in per_seed.values()])) for m in METRICS},
            conditional_group_bootstrap_95_intervals=intervals,
            positive_clipping_evidence_under_fixed_rule=bool(means['primary_score'] > 0
                and all(s['with_minus_without']['primary_score'] > 0 for s in per_seed.values())
                and intervals['primary_score'][0] > 0))
    return dict(status='complete', recipes=analyses, bootstrap_samples=bootstrap_samples, bootstrap_seed=20261126,
                independent_test_evidence=False, serving_selection_changed=False)


def cluster_intervals(pooled, groups, samples):
    import numpy as np
    if samples < 100:
        raise ValueError('At least 100 bootstrap draws required')
    names = sorted(set(groups.values())); gi = {g: i for i, g in enumerate(names)}
    languages = ('de', 'en', 'ja'); labels = ('negative', 'neutral', 'positive')
    counts = np.zeros((len(names), 3, 2, 3, 3, 3), dtype=np.int32)
    for si, seed in enumerate(SEEDS):
        for ai, arm in enumerate(('with', 'without')):
            for r in pooled[seed, arm]:
                counts[gi[groups[r['id']]], si, ai, languages.index(r['language']),
                       labels.index(r['reference']), labels.index(r['prediction'])] += 1
    draws = np.random.default_rng(20261126).multinomial(len(names), np.full(len(names), 1 / len(names)), size=samples)
    matrices = np.tensordot(draws, counts, axes=(1, 0))
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    denom = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0).mean(axis=-1).mean(axis=1)
    values = np.concatenate([f1[:, :, :2].mean(axis=-1, keepdims=True), f1], axis=-1)
    return {m: np.quantile(values[:, 0, i] - values[:, 1, i], [.025, .975]).tolist()
            for i, m in enumerate(METRICS)}
