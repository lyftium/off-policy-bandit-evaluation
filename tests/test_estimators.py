import numpy as np

from cbope.estimators import bootstrap_ci, estimate
from cbope.policies import LinUCB, one_hot
from cbope.replay import replay


def _synthetic(n=200_000, K=5, seed=0):
    rng = np.random.default_rng(seed)
    true = np.linspace(0.01, 0.05, K)
    a = rng.integers(0, K, n)
    r = (rng.random(n) < true[a]).astype(float)
    p = np.full(n, 1 / K)
    return true, a, r, p


def test_ipw_unbiased_for_deterministic_policy():
    true, a, r, p = _synthetic()
    dist = one_hot_k(np.full(len(a), 4), 5)
    for name in ["IPW", "SNIPW"]:
        assert abs(estimate(name, r, a, p, dist) - true[4]) < 0.004


def test_dr_with_perfect_model_matches_truth_and_ci_covers():
    true, a, r, p = _synthetic()
    dist = one_hot_k(np.full(len(a), 2), 5)
    q = np.tile(true, (len(a), 1))
    assert abs(estimate("DM", r, a, p, dist, q) - true[2]) < 1e-9
    pt, lo, hi = bootstrap_ci("DR", r, a, p, dist, q, n_boot=50)
    assert lo < pt < hi and abs(pt - true[2]) < 0.003 and hi - lo < 0.01


def test_linucb_learns_best_arm_in_replay():
    rng = np.random.default_rng(1)
    n, K = 60_000, 4
    x = np.ones((n, 1))
    a = rng.integers(0, K, n)
    r = (rng.random(n) < np.array([0.05, 0.1, 0.2, 0.6])[a]).astype(float)

    class L:  # minimal Logs-like stub
        action, reward = a, r
        def __len__(self):
            return n
    pol = LinUCB(d=1, n_arms=K, alpha=0.5)
    matched = replay(pol, L(), features=x)
    assert matched[-1000:].mean() > 0.45


def one_hot_k(choice, K):
    d = np.zeros((len(choice), K))
    d[np.arange(len(choice)), choice] = 1
    return d
