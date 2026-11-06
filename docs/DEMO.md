# Interview demo — off-policy evaluation

Walkthrough for a screener. Stick to artefacts in this repo.

## 90 seconds

1. **Question:** can we rank a new policy from another policy’s logs before an A/B?
2. Open the README “Start here” block, then the E1 table — every digit must match [`results.json`](../results.json).
3. Point at the CI badge and `pytest -q` (no data download).
4. **Limits out loud:** fashion e-commerce OBD, not aerospace; estimated lift ≠ observed A/B; full campaign needs the public zip (~15 min).

## 10 minutes

```bash
pip install -e ".[dev]"
pytest -q
# optional (~15 min + download):
# curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
# unzip open_bandit_dataset.zip -d data/
# PYTHONPATH=src python scripts/run_experiments.py --data data/open_bandit_dataset
```

| Minute | Show | Say |
|---|---|---|
| 0–2 | Problem + Start here | Score candidates from logs before A/B |
| 2–4 | E1 vs `results.json` | IPW / SNIPW / DR recover BTS CTR within 5.3–6.1 %; DM’s CI misses truth |
| 4–6 | E2 ranking | Popularity beats model-greedy; wide CI → A/B candidate, not shipped |
| 6–8 | Estimators + cross-fit DR | Importance weights + honest uncertainty |
| 8–9 | Eval-before-ship | Same habit as promoting any ML component |
| 9–10 | Limits | Only quote digits in `results.json` |

## Do not claim

- Aerospace or customer production bandit logs
- Digits not in `results.json`
- That DM’s tight CI is trustworthy here (it excludes the truth)
