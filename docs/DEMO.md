# Demo — off-policy evaluation

Inspect and reproduce an offline policy-evaluation campaign. Published values come from [`results.json`](../results.json), seed 0. Estimated lift is not an observed A/B outcome.

## Prerequisites

- Python 3.10+, a CPU, and pip
- Tests only: small dependency installation, no dataset
- Full campaign: `unzip`, the Open Bandit ZIP, sufficient RAM/disk for 1.37M rows, and about 15 minutes in the recorded environment

## Quick check (no data download)

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Expected: all unit/schema tests pass. CI runs these tests, not the full data campaign.

## Inspect published results

Open [`results.json`](../results.json) and compare the E1/E2/E3 tables. The README E1 values are direct transcriptions at reduced precision. Clicks quoted in prose are products of round count and CTR; they are not explicit result fields.

## Full campaign (optional)

The ZIP reports **412,931,917 bytes** via HTTP HEAD on November 6, 2026; expanded size varies by archive contents, so allow several GB. Check available disk space before downloading. The archive is supplied by the dataset publisher; inspect its contents and extract only the supplied public dataset:

```bash
python -m pip install -e ".[dev,experiments]"
curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip open_bandit_dataset.zip -d data/
PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset
```

Expected: a new `results.json` consistent with the published seed-0 campaign (environment/version differences can affect timings or numerical details). The current artifact records 880.5 seconds runtime. No dataset checksum or exact experiment dependency lock is published yet; keep the original ZIP and record package versions if reproducing for comparison.

## Additional results

### E2 — chronological policy holdout

Last 30%: 412,299 rounds; logger on-policy CTR = 0.00358963.

| Policy | SNIPW | DR (95% CI) |
|---|---:|---:|
| uniform (logger) | 0.0035896 | 0.0035755 [0.0034095, 0.0037651] |
| per-slot popularity | 0.0074488 | 0.0074187 [0.0051797, 0.0097142] |
| model-greedy (GBDT) | 0.0049981 | 0.0046309 [0.0026488, 0.0062633] |
| Bernoulli-TS (production) | 0.0048751 | 0.0048322 [0.0043243, 0.0053846] |

The chronological split is for policy learning/testing. The cross-fitted `q_all` reward-model folds are random; do not describe every model component as chronological.

### E3 — replay evaluation

| Learner | Matched rounds | CTR (95% CI) |
|---|---:|---:|
| random | 16,993 | 0.003649 [0.002706, 0.004708] |
| ε-greedy (ε=0.1) | 14,747 | 0.004747 [0.003526, 0.005967] |
| LinUCB α=0.1 | 12,044 | 0.004484 [0.003321, 0.005650] |
| LinUCB α=0.5 | 11,781 | 0.004499 [0.003310, 0.005772] |

Point estimates sit above random, but intervals overlap. Replay is data-hungry even when the logger is uniform.

## Limitations

- Public fashion e-commerce logs, not customer production data.
- Offline policy estimates are not live A/B outcomes.
- Candidate-policy support is limited and uncertainty is substantial.
- E1's independent reference is a logged CTR, not exact simulated truth.
