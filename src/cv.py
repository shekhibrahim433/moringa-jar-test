import numpy as np
from sklearn.model_selection import KFold

from .config import SEED


def lodo(groups):
    """Leave-one-group-out folds, ordered by group label."""
    groups = np.asarray(groups)
    return [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in np.unique(groups)]


def random_kfold(n, k=10, seed=SEED):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


def fold_ids(folds, n):
    out = np.full(n, -1)
    for i, (_, test) in enumerate(folds):
        out[test] = i
    if (out < 0).any():
        raise ValueError("some rows are never in a test fold")
    return out
