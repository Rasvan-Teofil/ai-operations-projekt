"""Pure probability aggregation; independent of model training and framework."""
import numpy as np


def aggregate(probabilities, chunks, mode='uniform'):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if mode == 'token_weighted':
        weights = np.asarray([max(1, sum(c['attention_mask'])-2) for c in chunks])
        return np.average(probabilities, axis=0, weights=weights).astype(np.float32)
    if mode != 'uniform':
        raise ValueError('Unknown aggregation mode')
    return probabilities.mean(0)
