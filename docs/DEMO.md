# Demo path (screener / interview)

No company names. Use only what this repository proves.

## 90 seconds

1. Problem: rank a new decision policy from another policy's logs before a live test.
2. Show top README table + E1 recovery numbers (must match [`results.json`](../results.json)).
3. Point at CI badge and offline `pytest` (no data download).
4. State limits: fashion e-commerce OBD logs, not aerospace; full campaign needs the public zip (~15 min).

## 10 minutes (laptop)

```bash
pip install -e ".[dev]"
pytest -q                                 # estimators / leakage / schema — no zip
# optional full campaign (~15 min + download):
# curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
# unzip open_bandit_dataset.zip -d data/
# PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset
```

Talk track while commands run:

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + Verify | Score candidates from logs before A/B |
| 2–4 | E1 table vs `results.json` | IPW/SNIPW/DR recover BTS CTR within 5.3–6.1%; DM CI misses truth |
| 4–6 | E2 policy ranking | Popularity beats model-greedy; wide CI → candidate for A/B, not shipped |
| 6–8 | Estimator code + cross-fit DR | Importance weights + honest uncertainty |
| 8–9 | Integrator angle | Same pattern as eval-before-deploy for ML components |
| 9–10 | Limits | Domain is OBD fashion logs; do not invent digits beyond `results.json` |

## Do not claim

- Aerospace or customer production bandit logs
- Digits not present in `results.json`
- That DM's tight CI is trustworthy here (it excludes the truth)
