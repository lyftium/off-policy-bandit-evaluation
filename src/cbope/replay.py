"""Replay evaluation (Li et al., 2011): unbiased for uniformly-random logs."""
from __future__ import annotations

import numpy as np

from .data import Logs


def replay(policy, logs: Logs, features: np.ndarray | None = None, seed: int = 0):
    """Stream logs in time order; keep only rounds where the policy agrees with
    the logger, and learn from them. Returns matched rewards (in order)."""
    feats = logs.x if features is None else features
    rewards = []
    for i in range(len(logs)):
        a = policy.select(feats[i])
        if a == logs.action[i]:
            r = logs.reward[i]
            policy.update(a, feats[i], r)
            rewards.append(r)
    return np.asarray(rewards)
