"""Reproduce every number and figure in the README.

    python scripts/run_experiments.py --data data/open_bandit_dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from cbope.data import N_ITEMS, N_POS, featurize, load_obd, time_split
from cbope.estimators import bootstrap_ci
from cbope.policies import EpsGreedy, LinUCB, model_greedy, popularity, uniform
from cbope.replay import replay
from cbope.reward_model import RewardModel, cross_fit_q


def bts_action_dist(position, seed):
    """Production Bernoulli-TS policy (ZOZOTOWN prior) as an (n, K) distribution."""
    from obp.policy import BernoulliTS
    bts = BernoulliTS(n_actions=N_ITEMS, len_list=N_POS, is_zozotown_prior=True,
                      campaign="all", random_state=seed)
    dist = bts.compute_batch_action_dist(n_rounds=1, n_sim=100_000)[0]  # (K, L)
    return dist.T.astype(np.float32)[position]


def on_policy_ctr(path):
    n = c = 0
    for ch in pd.read_csv(path, usecols=["click"], chunksize=2_000_000):
        n += len(ch)
        c += int(ch.click.sum())
    return c / n, n


def errbar(ax, labels, rows, truth=None, title="", rot=0):
    pts = np.array([r["estimate"] for r in rows]) * 100
    lo = np.array([r["ci_lo"] for r in rows]) * 100
    hi = np.array([r["ci_hi"] for r in rows]) * 100
    ax.errorbar(range(len(labels)), pts, yerr=[pts - lo, hi - pts], fmt="o", capsize=4)
    ax.set_xticks(range(len(labels)), labels, rotation=rot, fontsize=8)
    if truth is not None:
        ax.axhline(truth * 100, c="red", ls="--", lw=1, label="on-policy ground truth")
        ax.legend(fontsize=8)
    ax.set(title=title, ylabel="CTR (%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/open_bandit_dataset")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    (out / "figures").mkdir(exist_ok=True, parents=True)
    res = {"seed": a.seed, "n_boot": a.n_boot}
    t0 = time.time()

    df = load_obd(root / "random/all/all.csv", nrows=a.nrows)
    logs, _ = featurize(df)
    res["n_random_rounds"] = len(logs)
    res["random_ctr"] = float(logs.reward.mean())

    # EDA
    pos = df.groupby("position").click.mean()
    item = df.groupby("item_id").click.agg(["mean", "size"])
    res["eda"] = {"ctr_by_position": {int(k): float(v) for k, v in pos.items()},
                  "item_ctr_min": float(item["mean"].min()),
                  "item_ctr_median": float(item["mean"].median()),
                  "item_ctr_max": float(item["mean"].max()),
                  "days": int(df.timestamp.dt.date.nunique())}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].bar(pos.index.astype(str), pos * 100)
    ax[0].set(title="CTR by slot (uniform-random logs)", xlabel="position", ylabel="CTR (%)")
    s = item.sort_values("mean", ascending=False)
    se = np.sqrt(s["mean"] * (1 - s["mean"]) / s["size"])
    ax[1].errorbar(range(len(s)), s["mean"] * 100, yerr=1.96 * se * 100, fmt=".", ms=3, lw=0.6)
    ax[1].set(title="Item CTR with 95% CI (sorted)", xlabel="item rank", ylabel="CTR (%)")
    fig.tight_layout(); fig.savefig(out / "figures/eda.png", dpi=130); plt.close(fig)
    del df
    print("eda done", flush=True)

    # E1: estimator validation against an on-policy ground truth
    truth, n_bts = on_policy_ctr(root / "bts/all/all.csv")
    res["e1_ground_truth_ctr"], res["e1_bts_rounds"] = truth, n_bts
    dist_bts = bts_action_dist(logs.position, a.seed)
    q_all = cross_fit_q(logs, n_folds=2, seed=a.seed)
    e1 = {}
    for name in ["DM", "IPW", "SNIPW", "DR"]:
        pt, lo, hi = bootstrap_ci(name, logs.reward, logs.action, logs.pscore, dist_bts,
                                  q_all, n_boot=a.n_boot, seed=a.seed)
        e1[name] = {"estimate": pt, "ci_lo": lo, "ci_hi": hi,
                    "rel_error": abs(pt - truth) / truth, "covers_truth": bool(lo <= truth <= hi)}
    res["e1"] = e1
    print("E1", json.dumps(e1), flush=True)

    # E2: learn on first 70% of time, evaluate off-policy on the rest
    tr, te = time_split(logs, 0.7)
    q_pol = RewardModel(seed=a.seed).fit(tr).predict_all(te)
    p_logged = q_pol[np.arange(len(te)), te.action]
    res["reward_model_test"] = {
        "auc": float(roc_auc_score(te.reward, p_logged)),
        "log_loss": float(log_loss(te.reward, p_logged)),
        "base_rate_log_loss": float(log_loss(te.reward, np.full(len(te), tr.reward.mean())))}
    q_te = q_all[len(tr):]
    pols = {"uniform (logger)": uniform(te), "popularity": popularity(tr, te),
            "model-greedy (GBDT)": model_greedy(q_pol),
            "Bernoulli-TS (production)": bts_action_dist(te.position, a.seed)}
    e2 = {}
    for p, dist in pols.items():
        e2[p] = {}
        for name in ["SNIPW", "DR"]:
            pt, lo, hi = bootstrap_ci(name, te.reward, te.action, te.pscore, dist, q_te,
                                      n_boot=a.n_boot, seed=a.seed)
            e2[p][name] = {"estimate": pt, "ci_lo": lo, "ci_hi": hi}
    res["e2_test_rounds"] = len(te)
    res["e2_test_on_policy_ctr"] = float(te.reward.mean())
    res["e2"] = e2
    print("E2", json.dumps(e2), flush=True)

    # E3: online learners via replay
    d = logs.x.shape[1]
    learners = {"random": EpsGreedy(eps=1.0, seed=a.seed),
                "eps-greedy 0.1": EpsGreedy(eps=0.1, seed=a.seed),
                "LinUCB a=0.1": LinUCB(d, alpha=0.1), "LinUCB a=0.5": LinUCB(d, alpha=0.5)}
    rng = np.random.default_rng(a.seed)
    e3, curves = {}, {}
    for name, pol in learners.items():
        r = replay(pol, logs)
        bs = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(a.n_boot)]
        e3[name] = {"matched_rounds": int(len(r)), "estimate": float(r.mean()),
                    "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))}
        curves[name] = np.cumsum(r) / np.arange(1, len(r) + 1)
        print(name, e3[name], flush=True)
    res["e3"] = e3

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    errbar(ax[0], list(e1), list(e1.values()), truth, "E1: estimating production BTS from random logs")
    errbar(ax[1], list(e2), [v["DR"] for v in e2.values()], None, "E2: DR value on held-out days", rot=15)
    for k, c in curves.items():
        ax[2].plot(c[200:] * 100, label=k, lw=1)
    ax[2].set(title="E3: replay cumulative CTR", xlabel="matched rounds", ylabel="CTR (%)")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures/results.png", dpi=130); plt.close(fig)

    res["runtime_sec"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("done", res["runtime_sec"])


if __name__ == "__main__":
    main()
