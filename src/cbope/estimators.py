"""Off-policy value estimators (DM, IPW, SNIPW, DR) with bootstrap CIs.

Notation: n logged rounds, K actions. ``action_dist[i, a]`` is the evaluation
policy's probability of choosing ``a`` in round i; ``q_hat[i, a]`` is a reward
model's estimate of E[r | x_i, a].
"""
from __future__ import annotations

import numpy as np


def _w(action, pscore, action_dist):
    return action_dist[np.arange(len(action)), action] / pscore


def round_dm(reward, action, pscore, action_dist, q_hat):
    return (action_dist * q_hat).sum(1)


def round_ipw(reward, action, pscore, action_dist, q_hat=None):
    return _w(action, pscore, action_dist) * reward


def round_dr(reward, action, pscore, action_dist, q_hat):
    idx = np.arange(len(action))
    w = _w(action, pscore, action_dist)
    return (action_dist * q_hat).sum(1) + w * (reward - q_hat[idx, action])


def snipw(reward, action, pscore, action_dist, q_hat=None):
    w = _w(action, pscore, action_dist)
    return float((w * reward).sum() / w.sum())


ROUND = {"DM": round_dm, "IPW": round_ipw, "DR": round_dr}


def estimate(name, reward, action, pscore, action_dist, q_hat=None):
    if name == "SNIPW":
        return snipw(reward, action, pscore, action_dist)
    return float(ROUND[name](reward, action, pscore, action_dist, q_hat).mean())


def bootstrap_ci(name, reward, action, pscore, action_dist, q_hat=None,
                 n_boot=200, alpha=0.05, seed=0):
    """Percentile bootstrap over logged rounds. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(reward)
    point = estimate(name, reward, action, pscore, action_dist, q_hat)
    if name == "SNIPW":
        w = _w(action, pscore, action_dist)
        wr = w * reward
        stats = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            stats.append(wr[i].sum() / w[i].sum())
    else:
        per = ROUND[name](reward, action, pscore, action_dist, q_hat)
        stats = [per[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
