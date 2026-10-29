"""Click model q(x, a) used by DM/DR and by the model-greedy policy."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import N_ITEMS, Logs


def design(x: np.ndarray, affinity: np.ndarray, action: np.ndarray) -> np.ndarray:
    idx = np.arange(len(action))
    return np.hstack([x, action[:, None].astype(np.float32), affinity[idx, action][:, None]])


class RewardModel:
    def __init__(self, seed: int = 0, **kw):
        params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
                      l2_regularization=1.0, random_state=seed)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)

    def fit(self, logs: Logs) -> "RewardModel":
        X = design(logs.x, logs.affinity, logs.action)
        # action id column treated as categorical
        self.clf.set_params(categorical_features=[X.shape[1] - 2])
        self.clf.fit(X, logs.reward.astype(int))
        return self

    def predict_all(self, logs: Logs, batch: int = 200_000) -> np.ndarray:
        """(n, K) matrix of predicted click probabilities for every item."""
        n = len(logs)
        out = np.empty((n, N_ITEMS), dtype=np.float32)
        for a in range(N_ITEMS):
            acts = np.full(n, a)
            out[:, a] = self.clf.predict_proba(design(logs.x, logs.affinity, acts))[:, 1]
        return out


def cross_fit_q(logs: Logs, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Out-of-fold q_hat so DR's reward model never sees the row it corrects."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(logs))
    q = np.empty((len(logs), N_ITEMS), dtype=np.float32)
    for k in range(n_folds):
        m = RewardModel(seed=seed).fit(logs.subset(fold != k))
        q[fold == k] = m.predict_all(logs.subset(fold == k))
    return q
