# Off-policy evaluation on 1.37 M real bandit logs

![CI](https://github.com/lyftium/off-policy-bandit-evaluation/actions/workflows/ci.yml/badge.svg)

Can you rank a new recommendation policy from another policy’s logs **before** an A/B test?

On the Open Bandit Dataset the answer is yes, within **5.3–6.1 %** relative error. IPW / SNIPW / DR estimates from uniform-random logs land near the production Bernoulli-TS CTR, and their 95 % bootstrap CIs cover it. Direct Method’s CI is tiny and misses the truth. With the honest estimators, a per-slot popularity policy looks about **2.1×** the logger on a chronological hold-out — worth an A/B, not a ship decision by itself.

**Start here:** [`docs/DEMO.md`](docs/DEMO.md) — every number below is in [`results.json`](results.json) (seed 0).

Python 3.10+, a GBDT reward model, DM / IPW / SNIPW / DR with 200 bootstrap CIs. Domain is fashion e-commerce OBD, not aerospace; estimated lift is not an observed A/B.

[Data](#data) · [Method](#method) · [Results](#results) · [Takeaways](#takeaways) · [Reproduce](#reproduce)

## Data

[Open Bandit Dataset](https://research.zozo.com/data_release/open_bandit_dataset.zip) (Saito et al., NeurIPS Datasets & Benchmarks 2021; ZOZOTOWN; CC BY 4.0). Two campaigns ran over the same seven days:

| Log | Policy | Rounds | Clicks | CTR |
|---|---|---:|---:|---:|
| `random/all` | uniform over 80 items × 3 slots (propensity 1/80) | 1,374,327 | 4,768 | 0.347% |
| `bts/all` | production Bernoulli Thompson Sampling | 12,357,200 | 61,208 | 0.495% |

Context: four categorical user features (one-hot, 24 dims) + slot; 80 user–item affinity scores.

![EDA](figures/eda.png)

Clicks are rare (0.35%), so variance is the main enemy. The slot effect is small; item CTRs span 0.115%–0.784% (median 0.332%).

## Method

- **Estimators** (`src/cbope/estimators.py`) — DM, IPW, self-normalised IPW, Doubly Robust; 95 % percentile bootstrap (200 resamples).
- **Reward model** (`reward_model.py`) — GBDT on context + item id + affinity, **2-fold cross-fitted** so DR never corrects a row it trained on.
- **Policies** (`policies.py`) — uniform, per-slot smoothed popularity, model-greedy, production BTS, ε-greedy, disjoint LinUCB (Sherman–Morrison).

Three experiments:

1. **E1 — estimator check.** Estimate BTS value from random logs only; compare to on-policy BTS CTR.
2. **E2 — policy selection.** Chronological 70/30; learn on the first 70%, evaluate off-policy on the last 412,299 rounds.
3. **E3 — online learners.** Replay evaluation (Li et al., 2011) — unbiased because the logger is uniform.

## Results

Every number below comes from `scripts/run_experiments.py` (seed 0) → [`results.json`](results.json).

### E1 — recover a known truth

Ground truth = 0.4953% (BTS on-policy CTR).

| Estimator | Estimate | 95% CI | Rel. error | CI covers truth |
|---|---:|---|---:|:-:|
| DM | 0.4649% | [0.4647, 0.4651] | 6.1% | ✗ |
| IPW | 0.4692% | [0.4382, 0.5022] | 5.3% | ✓ |
| SNIPW | 0.4692% | [0.4393, 0.5015] | 5.3% | ✓ |
| DR | 0.4667% | [0.4366, 0.4990] | 5.8% | ✓ |

DM is precise but over-confident: the classic failure of a model-only estimate with no importance-weight correction.

### E2 — which policy would you ship?

Held-out last 30%; logger CTR = 0.3590%.

| Policy | SNIPW | DR (95% CI) |
|---|---:|---|
| uniform (logger) | 0.3590% | 0.3575% [0.3410, 0.3765] |
| **per-slot popularity** | **0.7449%** | **0.7419% [0.5180, 0.9714]** |
| model-greedy (GBDT) | 0.4998% | 0.4631% [0.2649, 0.6263] |
| Bernoulli-TS (production) | 0.4875% | 0.4832% [0.4324, 0.5385] |

The reward model barely beats the base rate on held-out days (AUC 0.626), so greedy argmax over it is noisy and the simpler popularity policy wins. Its CI is wide (~1/80 of logged rounds match). Treat it as an A/B candidate, not a shipped result.

### E3 — online learners

Replay over 1.37 M rounds.

| Learner | Matched rounds | CTR (95% CI) |
|---|---:|---|
| random | 16,993 | 0.365% [0.271, 0.471] |
| ε-greedy (ε=0.1) | 14,747 | 0.475% [0.353, 0.597] |
| LinUCB α=0.1 | 12,044 | 0.448% [0.332, 0.565] |
| LinUCB α=0.5 | 11,781 | 0.450% [0.331, 0.577] |

Point estimates sit above random; CIs overlap. Replay is unbiased but data-hungry — which is why E1–E2 matter.

![results](figures/results.png)

## Takeaways

1. Off-policy estimates on real logs came within 5–6% of an independent ground truth. Report CIs, not point estimates alone.
2. Cross-fitted DR ≈ SNIPW here: uniform logging keeps weights bounded (≤ 80), so DR’s variance reduction is modest.
3. With weak features (AUC 0.63), a smoothed popularity baseline beats a GBDT-greedy policy.

## Reproduce

```bash
pip install -e ".[dev,experiments]"
curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip && unzip open_bandit_dataset.zip -d data/
PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset   # ~15 min
pytest -q                                                                           # no data needed
```

`obp` is used only for the production BTS action distribution; estimators, policies and models live in this repo.

```
src/cbope/   data.py · estimators.py · reward_model.py · policies.py · replay.py
scripts/     run_experiments.py
tests/       synthetic unbiasedness, oracle DR, LinUCB, leakage-free split
figures/     eda.png · results.png
releases/    example-obd-seed0.json
```

## References

Saito et al. (2021) *Open Bandit Dataset and Pipeline*. Li et al. (2010) *A Contextual-Bandit Approach to Personalized News Article Recommendation*. Li et al. (2011) *Unbiased Offline Evaluation…*. Dudík et al. (2011) *Doubly Robust Policy Evaluation and Learning*.

MIT
