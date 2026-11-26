"""Dependency-free experimental design and paired descriptive contrasts."""
from itertools import combinations, product
import random

SEED = 20261110
FACTORS = {
    'gradient_clip': [None, 1.0],
    'learning_rate': [5e-5, 1e-4],
    'weight_decay': [0.0, 0.01],
    'warmup_fraction': [0.0, 0.1],
}
REFERENCE = dict(gradient_clip=1.0, learning_rate=1e-4, weight_decay=0.01, warmup_fraction=0.1)


def configurations():
    grid = [dict(zip(FACTORS, values)) for values in product(*FACTORS.values())]
    reference = grid.pop(grid.index(REFERENCE))
    random.Random(20261126).shuffle(grid)
    rows = []
    for seed in (SEED, SEED + 1):
        others = list(grid)
        random.Random(seed).shuffle(others)
        rows.extend(dict(row, seed=seed, block='factorial') for row in [reference, *others])
    return [dict(row, id=f'run-{i:02d}') for i, row in enumerate(rows, 1)]



def validate_membership(table_rows, member):
    """Reject missing/duplicate IDs, overlapping groups and incomplete partitions."""
    by_id = {r['id']: r for r in table_rows}
    fit, dev = member['fit_ids'], member['dev_ids']
    if (len(by_id) != len(table_rows) or len(set(fit)) != len(fit)
            or len(set(dev)) != len(dev) or set(fit) & set(dev)
            or set(fit) | set(dev) != set(by_id)):
        raise ValueError('Membership must partition the unique Train IDs exactly')
    if {by_id[i]['group'] for i in fit} & {by_id[i]['group'] for i in dev}:
        raise ValueError('Fit/development groups overlap')
    for ids in (fit, dev):
        support = {(by_id[i]['language'], by_id[i]['label']) for i in ids}
        if len(support) != 9:
            raise ValueError('Every language/class needs support in both partitions')


def contrasts(results, metric='primary_score'):
    """Compute the complete factorial separately by seed, then average effects."""
    per_seed = {}
    for seed in (SEED, SEED + 1):
        grid = [r for r in results if r['config']['seed'] == seed]
        if len(grid) != 16:
            return {'status': 'pending', 'completed_runs': len(results), 'expected_runs': 32}
        per_seed[str(seed)] = _seed_contrasts(grid, metric)
    effects = {}
    for name in FACTORS:
        parts = [r['main_effects'][name] for r in per_seed.values()]
        effects[name] = dict(parts[0], high_minus_low=sum(p['high_minus_low'] for p in parts)/2,
                            paired_differences=[d for p in parts for d in p['paired_differences']], pairs=16)
    return {'status': 'complete', 'metric': metric, 'main_effects': effects, 'per_seed': per_seed,
            'two_way_difference_of_differences': {
                name: sum(r['two_way_difference_of_differences'][name] for r in per_seed.values())/2
                for name in next(iter(per_seed.values()))['two_way_difference_of_differences']},
            'interpretation': 'Descriptive fixed-split effects with two seeds; pairs share data. No p-values or causal mechanism proof.'}


def _seed_contrasts(grid, metric):
    lookup = {tuple(r['config'][k] for k in FACTORS): r['metrics'][metric] for r in grid}
    if len(lookup) != 16:
        raise ValueError('Duplicate or missing factorial condition')
    names = list(FACTORS)
    effects = {}
    for name in names:
        other = [n for n in names if n != name]
        diffs = []
        for values in product(*(FACTORS[n] for n in other)):
            fixed = dict(zip(other, values))
            scores = []
            for level in FACTORS[name]:
                condition = dict(fixed, **{name: level})
                scores.append(lookup[tuple(condition[n] for n in names)])
            diffs.append(scores[1] - scores[0])
        effects[name] = {'high_minus_low': sum(diffs) / len(diffs), 'paired_differences': diffs,
                         'low': FACTORS[name][0], 'high': FACTORS[name][1], 'pairs': len(diffs)}
    interactions = {}
    for first, second in combinations(names, 2):
        other = [n for n in names if n not in (first, second)]
        diffs = []
        for values in product(*(FACTORS[n] for n in other)):
            fixed = dict(zip(other, values))
            scores = {}
            for i, j in product(range(2), repeat=2):
                condition = dict(fixed, **{first: FACTORS[first][i], second: FACTORS[second][j]})
                scores[i, j] = lookup[tuple(condition[n] for n in names)]
            diffs.append(scores[1, 1] - scores[0, 1] - scores[1, 0] + scores[0, 0])
        interactions[f'{first} × {second}'] = sum(diffs) / len(diffs)
    return {'status': 'complete', 'metric': metric, 'main_effects': effects,
            'two_way_difference_of_differences': interactions,
            'interpretation': 'Descriptive fixed-split effects, not independent samples or causal mechanism proof.'}
