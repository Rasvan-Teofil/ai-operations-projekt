import numpy as np
from src.second_path.threshold_curve import negative_curve


def test_lower_threshold_trades_misses_for_false_alarms():
    # truth: 0=negative, 1=neutral, 2=positive
    proba = np.array([[.6, .3, .1], [.4, .5, .1], [.3, .6, .1], [.1, .2, .7]])
    truth = [0, 0, 1, 2]
    c = negative_curve(proba, truth, margins=[0.0, 0.15, 0.35])
    assert list(c.caught) == [1, 2, 2]
    assert list(c.missed) == [1, 0, 0]
    assert list(c.false_alarms) == [0, 0, 1]
    assert c.recall_negative.is_monotonic_increasing
