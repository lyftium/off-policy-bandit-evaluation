"""Evaluation policies. Each returns an (n, K) action distribution."""
from __future__ import annotations

import numpy as np

from .data import N_ITEMS, N_POS, Logs


def uniform(logs: Logs) -> np.ndarray:
    return np.full((len(logs), N_ITEMS), 1.0 / N_ITEMS, dtype=np.float32)


def one_hot(choice: np.ndarray) -> np.ndarray:
    d = np.zeros((len(choice), N_ITEMS), dtype=np.float32)
    d[np.arange(len(choice)), choice] = 1.0
    return d


def popularity(train: Logs, test: Logs, prior: float = 1.0) -> np.ndarray:
    """Greedy on per-(position, item) CTR with a Beta smoothing prior."""
    base = train.reward.mean()
    best = np.empty(N_POS, dtype=int)
    for p in range(N_POS):
        m = train.position == p
        clicks = np.bincount(train.action[m], train.reward[m], N_ITEMS)
        imps = np.bincount(train.action[m], minlength=N_ITEMS)
        best[p] = np.argmax((clicks + prior * base * 100) / (imps + prior * 100))
    return one_hot(best[test.position])


def model_greedy(q_test: np.ndarray) -> np.ndarray:
    return one_hot(q_test.argmax(1))


class LinUCB:
    """Disjoint LinUCB (Li et al., 2010) with per-arm ridge regression."""

    def __init__(self, d: int, n_arms: int = N_ITEMS, alpha: float = 1.0, lam: float = 1.0):
        self.alpha = alpha
        self.A_inv = np.repeat(np.eye(d)[None] / lam, n_arms, 0)
        self.b = np.zeros((n_arms, d))

    def select(self, x: np.ndarray) -> int:
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        mean = theta @ x
        var = np.einsum("i,kij,j->k", x, self.A_inv, x)
        return int(np.argmax(mean + self.alpha * np.sqrt(var)))

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        Ai = self.A_inv[a]
        Ax = Ai @ x
        self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)  # Sherman-Morrison
        self.b[a] += r * x


class EpsGreedy:
    def __init__(self, n_arms: int = N_ITEMS, eps: float = 0.1, seed: int = 0):
        self.eps, self.rng = eps, np.random.default_rng(seed)
        self.c, self.n = np.zeros(n_arms), np.zeros(n_arms)

    def select(self, x) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(len(self.c)))
        return int(np.argmax((self.c + 0.01) / (self.n + 1)))

    def update(self, a, x, r):
        self.c[a] += r
        self.n[a] += 1
