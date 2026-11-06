# Off-policy evaluation benchmark

## Question

Can I evaluate a candidate recommendation policy using logs from another policy before spending traffic on an A/B test? I measure estimator accuracy and uncertainty on public Open Bandit data rather than treating an offline point estimate as a realized uplift.

The complete measured campaign is in [`results.json`](results.json). The implementation and exact experiment flow are in [`scripts/run_experiments.py`](scripts/run_experiments.py). Figures are in [`figures/`](figures/).

## Method and reproduction

- **Data:** Open Bandit Dataset; `random/all` has 1,374,327 uniform-random rounds with observed CTR 0.0034693344; `bts/all` has 12,357,200 rounds with observed CTR 0.0049532256.
- **E1:** estimate BTS policy value from the separate uniform-random log using DM, IPW, SNIPW, and doubly robust (DR), then compare with the independently logged BTS CTR.
- **E2:** learn on the first 70% and evaluate on the final 412,299 chronological rounds. Candidates include per-slot popularity and a model-greedy policy.
- **E3:** replay evaluation of ε-greedy and LinUCB on the uniform log.
- **Uncertainty:** 200 percentile bootstrap resamples, seed 0. The reward model is two-fold cross-fitted. Recorded campaign runtime: 880.5 seconds.

Run the seed-0 campaign after inspecting the dataset download and installing the experiment dependencies:

```bash
python -m pip install -e ".[dev,experiments]"
curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip open_bandit_dataset.zip -d data/
PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset
```

The full dataset campaign was not re-run for this publication. Public raw results preserve the measured run.

## Estimator validation (E1)

| Estimator | Estimate | 95% bootstrap interval | Relative error | Covers logged BTS reference |
|---|---:|---:|---:|:---:|
| DM | 0.0046488242 | [0.0046472094, 0.0046506684] | 0.0614552 | No |
| IPW | 0.0046919879 | [0.0043817785, 0.0050221995] | 0.0527409 | Yes |
| SNIPW | 0.0046918482 | [0.0043933221, 0.0050145840] | 0.0527691 | Yes |
| DR | 0.0046667640 | [0.0043658894, 0.0049896832] | 0.0578334 | Yes |

The reference is a finite logged CTR, not error-free simulated ground truth. The DM interval misses it; the other intervals cover it. A narrow interval alone is not evidence of accuracy.

## Chronological candidate evaluation (E2)

| Candidate | DR estimate | 95% bootstrap interval |
|---|---:|---:|
| Uniform logger | 0.0035754568 | [0.0034095377, 0.0037650814] |
| Per-slot popularity | 0.0074187275 | [0.0051796771, 0.0097141861] |
| Model-greedy (GBDT) | 0.0046308559 | [0.0026488497, 0.0062632914] |
| Production Bernoulli-TS | 0.0048322489 | [0.0043243128, 0.0053845525] |

Held-out logger on-policy CTR is 0.0035896279. The reward model's held-out AUC is 0.6257459. Popularity is a candidate for an A/B test, not an observed improvement. Low support for exact target slot-item matches makes these estimates uncertain.

## Replay experiment (E3)

| Policy | Matched rounds | CTR estimate | 95% interval |
|---|---:|---:|---:|
| Random | 16,993 | 0.0036485612 | [0.0027055258, 0.0047078209] |
| ε-greedy 0.1 | 14,747 | 0.0047467281 | [0.0035261409, 0.0059673154] |
| LinUCB a=0.1 | 12,044 | 0.0044835603 | [0.0033211558, 0.0056501162] |
| LinUCB a=0.5 | 11,781 | 0.0044987692 | [0.0033104151, 0.0057720058] |

Overlapping intervals do not support a reliable winner.

## Limitations

Public fashion-click logs only; no live experiment or realized lift. Results depend on logging and propensity assumptions. The chronological split applies to policy learning/evaluation, not the separate random cross-fitted reward-model folds. The artifact does not pin a dataset checksum or complete dependency lock. CPU/schema tests do not rerun the 1.37M-row campaign.
