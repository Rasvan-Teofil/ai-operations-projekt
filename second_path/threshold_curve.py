"""Recall/precision for `negative` when its decision threshold is lowered.

Lever from docs/error-costs.md: flag a text as negative when
p(negative) + margin >= max(p(neutral), p(positive)), without retraining.

The margin is chosen on the validation split only. The test split stays
locked by the evaluation protocol (reports/evaluation.lock), so test numbers
for a chosen margin are reported once, never used for the choice.

Usage (needs models/second_path/finetuned and data/second_path/processed):
    python -m second_path.threshold_curve            # validation split
"""
import argparse
import numpy as np
import pandas as pd
from second_path.config import DATA, MODELS, REPORTS, LABELS

NEG = LABELS.index('negative')
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def negative_curve(proba, truth, margins=MARGINS):
    """proba: (n, 3) class probabilities; truth: label indices."""
    proba = np.asarray(proba, dtype=float)
    truth = np.asarray(truth)
    others = np.delete(proba, NEG, axis=1).max(axis=1)
    is_neg = truth == NEG
    rows = []
    for m in margins:
        flagged = proba[:, NEG] + m >= others
        tp = int((flagged & is_neg).sum())
        fp = int((flagged & ~is_neg).sum())
        fn = int((~flagged & is_neg).sum())
        rows.append({'margin': m, 'flagged': int(flagged.sum()), 'caught': tp,
                     'missed': fn, 'false_alarms': fp,
                     'recall_negative': tp / max(tp + fn, 1),
                     'precision_negative': tp / max(tp + fp, 1),
                     'false_alarms_per_miss': fp / max(fn, 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    from second_path.inference import Predictor
    val = pd.read_csv(DATA / 'val.csv')
    truth = [LABELS.index(v) for v in val.label]
    predictor = Predictor(MODELS / 'finetuned', device=args.device)
    proba = np.vstack([predictor.predict_proba(list(val.text[i:i + 64]))
                       for i in range(0, len(val), 64)])
    curve = negative_curve(proba, truth)
    out = REPORTS / 'negative_threshold_curve_val.csv'
    curve.to_csv(out, index=False)
    print(curve.to_string(index=False, float_format='%.3f'))
    print(f'written: {out}')


if __name__ == '__main__':
    main()
