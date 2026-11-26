"""Fixed paired design and pooled out-of-fold analysis, without model inference."""
from itertools import product
import random

SEEDS = (20261110, 20261111, 20261112)
RECIPES = {
    'baseline': dict(gradient_clip=1.0, learning_rate=1e-4, weight_decay=0.01, warmup_fraction=0.1),
    'candidate': dict(gradient_clip=1.0, learning_rate=5e-5, weight_decay=0.0, warmup_fraction=0.0),
    'candidate_no_clip': dict(gradient_clip=None, learning_rate=5e-5, weight_decay=0.0, warmup_fraction=0.0),
}
REUSE = {(0, 20261110, 'baseline'): 'run-01', (0, 20261111, 'baseline'): 'run-17',
         (0, 20261110, 'candidate'): 'run-06', (0, 20261111, 'candidate'): 'run-26',
         (0, 20261110, 'candidate_no_clip'): 'run-09', (0, 20261111, 'candidate_no_clip'): 'run-27'}


def configurations():
    pairs = list(product(range(3), SEEDS)); random.Random(20261126).shuffle(pairs)
    rows = []
    for i, (fold, seed) in enumerate(pairs):
        order = list(RECIPES); random.Random(20261126+i).shuffle(order)
        for recipe in order:
            rows.append(dict(id=f'f{fold}-s{seed}-{recipe}', fold=fold, seed=seed,
                             recipe=recipe, **RECIPES[recipe]))
    return rows


def validate_folds(rows, members):
    from src.studies.optimizer_ablation.design import validate_membership
    if sorted(m['fold'] for m in members) != [0, 1, 2]:
        raise ValueError('Exactly three distinct grouped folds required')
    held = []
    for member in members:
        validate_membership(rows, member)
        held.extend(member['dev_ids'])
    if len(held) != len(set(held)) or set(held) != {r['id'] for r in rows}:
        raise ValueError('Out-of-fold predictions must partition all Train IDs exactly once')


def score(records):
    from sklearn.metrics import accuracy_score, f1_score
    labels = ['negative', 'neutral', 'positive']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('OOF records must have unique article IDs')
    values = {}
    for language in ('de', 'en', 'ja'):
        subset = [r for r in records if r['language'] == language]
        if not subset:
            raise ValueError('Missing language in OOF records')
        values['macro_f1_'+language] = float(f1_score([r['reference'] for r in subset], [r['prediction'] for r in subset], labels=labels, average='macro', zero_division=0))
    values['primary_score'] = (values['macro_f1_de'] + values['macro_f1_en'])/2
    values['accuracy'] = float(accuracy_score([r['reference'] for r in records], [r['prediction'] for r in records]))
    return values


def paired_analysis(results, predictions, expected_ids, groups, bootstrap_samples=2000):
    import numpy as np
    expected = {c['id']: c for c in configurations()}
    if len(results) != 27 or {r['config']['id'] for r in results} != set(expected):
        return dict(status='pending', completed=len(results), expected=27)
    for row in results:
        if row['config'] != expected[row['config']['id']]:
            raise ValueError('Changed paired configuration')
    metrics = ['primary_score', 'macro_f1_de', 'macro_f1_en', 'macro_f1_ja']
    comparisons = {'recipe': ('candidate', 'baseline'), 'clipping': ('candidate', 'candidate_no_clip')}
    pooled, per_seed = {}, {}
    metadata = None
    for seed in SEEDS:
        pooled[seed] = {}; per_seed[str(seed)] = {}
        scored, reduced = {}, {}
        for recipe in RECIPES:
            records = [r for fold in range(3) for r in predictions[f'f{fold}-s{seed}-{recipe}']]
            if len(records) != len(expected_ids) or {r['id'] for r in records} != set(expected_ids):
                raise ValueError('OOF coverage is not exactly the full Train set')
            current = {r['id']:(r['reference'], r['language'], r['text_sha256']) for r in records}
            if metadata is not None and current != metadata:
                raise ValueError('Paired reference metadata differs')
            metadata = current
            pooled[seed][recipe] = records
            scored[recipe] = score(records)
            reduced[recipe] = score([r for fold in (1,2) for r in predictions[f'f{fold}-s{seed}-{recipe}']])
        per_seed[str(seed)] = dict(oof=scored, excluding_selection_fold_oof=reduced,
            contrasts={name:{m:scored[a][m]-scored[b][m] for m in metrics} for name,(a,b) in comparisons.items()},
            excluding_selection_fold_contrasts={name:{m:reduced[a][m]-reduced[b][m] for m in metrics} for name,(a,b) in comparisons.items()})
    means = {name:{m:float(np.mean([v['contrasts'][name][m] for v in per_seed.values()])) for m in metrics} for name in comparisons}
    outside = {name:{m:float(np.mean([v['excluding_selection_fold_contrasts'][name][m] for v in per_seed.values()])) for m in metrics} for name in comparisons}
    intervals = cluster_intervals(pooled, groups, bootstrap_samples)
    recipe_gain=means['recipe']; clip_gain=means['clipping']
    gates = dict(mean_primary_gain_at_least_one_pp=recipe_gain['primary_score']>=0.01,
        nonnegative_primary_each_seed=all(v['contrasts']['recipe']['primary_score']>=0 for v in per_seed.values()),
        no_mean_language_regression_over_one_pp=all(recipe_gain['macro_f1_'+lang]>=-0.01 for lang in ('de','en','ja')))
    clipping_supported=(clip_gain['primary_score']>0 and all(v['contrasts']['clipping']['primary_score']>0 for v in per_seed.values())
        and intervals['clipping']['primary_score'][0]>0)
    return dict(status='complete', per_seed=per_seed, mean_contrasts=means,
        excluding_selection_fold_mean_contrasts=outside, conditional_group_bootstrap_95_intervals=intervals,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=20261126, practical_recipe_gate=gates,
        clipping_supported_in_this_exploratory_cv=bool(clipping_supported),
        recommendation='candidate_merits_fresh_independent_evaluation' if all(gates.values()) else 'retain_current_default',
        independent_test_evidence=False, serving_selection_changed=False,
        limits='Post-selection exploratory CV on reused Train data. Seeds/folds share articles. Group bootstrap is conditional on these trained fits, fixed seeds and selected recipes; it does not correct selection bias or establish independent-Test improvement. No final-Test inference.')


def cluster_intervals(pooled, groups, samples):
    """Paired cluster resampling: whole semantic groups, same draw for all arms/seeds."""
    import numpy as np
    if samples < 100: raise ValueError('At least 100 bootstrap draws required')
    group_names=sorted(set(groups.values())); gi={g:i for i,g in enumerate(group_names)}
    langs=('de','en','ja'); labels=('negative','neutral','positive'); recipes=list(RECIPES)
    counts=np.zeros((len(group_names),3,3,3,3,3),dtype=np.int32)
    for si,seed in enumerate(SEEDS):
        for ai,recipe in enumerate(recipes):
            for row in pooled[seed][recipe]:
                counts[gi[groups[row['id']]],si,ai,langs.index(row['language']),labels.index(row['reference']),labels.index(row['prediction'])]+=1
    rng=np.random.default_rng(20261126); draws=rng.multinomial(len(group_names),np.full(len(group_names),1/len(group_names)),size=samples)
    matrices=np.tensordot(draws,counts,axes=(1,0))
    tp=np.diagonal(matrices,axis1=-2,axis2=-1)
    denominator=matrices.sum(axis=-1)+matrices.sum(axis=-2)
    f1=np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator>0).mean(axis=-1)
    # samples x seed x arm x language; average F1 across the three fixed seeds.
    mean=f1.mean(axis=1); values=np.concatenate([mean[:,:,:2].mean(axis=-1,keepdims=True),mean],axis=-1)
    names=('primary_score','macro_f1_de','macro_f1_en','macro_f1_ja')
    return {name:{metric:np.quantile(values[:,recipes.index(a),i]-values[:,recipes.index(b),i],[.025,.975]).tolist()
                  for i,metric in enumerate(names)}
            for name,(a,b) in {'recipe':('candidate','baseline'),'clipping':('candidate','candidate_no_clip')}.items()}
