# Demo — off-policy evaluation

Reproduce the offline tests and inspect published results. No A/B claim; numbers come from [`results.json`](../results.json) (seed 0).

## Prerequisites

- Python 3.10+
- ~few minutes for unit tests; ~15 min + download for the full Open Bandit campaign

## Quick check (no data download)

```bash
pip install -e ".[dev]"
pytest -q
```

**Expected:** all tests pass.

## Inspect published results

Open [`results.json`](../results.json) and the README E1 / E2 tables. Digits should match.

## Full campaign (optional)

```bash
curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip open_bandit_dataset.zip -d data/
PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset
```

**Expected:** a new `results.json` consistent with the published seed-0 campaign (allow for environment noise on wall time only).

## Limitations

- Fashion e-commerce Open Bandit logs — not aerospace or customer production data
- Estimated policy lift ≠ observed A/B outcome
- Full campaign needs the public zip
