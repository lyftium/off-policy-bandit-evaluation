"""Guard published results.json shape — no invented metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_results_json_has_core_experiment_keys():
    data = json.loads((ROOT / "results.json").read_text())
    for key in ("seed", "e1", "e2", "e3", "e1_ground_truth_ctr"):
        assert key in data, f"missing {key}"
    assert data["seed"] == 0
    assert isinstance(data["e1"], dict) and len(data["e1"]) >= 1
    assert isinstance(data["e2"], dict) and len(data["e2"]) >= 1
    assert isinstance(data["e3"], dict) and len(data["e3"]) >= 1


def test_e1_estimators_present():
    e1 = json.loads((ROOT / "results.json").read_text())["e1"]
    names = {k.lower() for k in e1}
    for need in ("ipw", "snipw", "dr", "dm"):
        assert any(need in n for n in names), f"missing estimator-like key for {need} in {names}"
