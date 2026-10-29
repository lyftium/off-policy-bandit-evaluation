"""Loading and featurising Open Bandit Dataset (OBD) logs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

USER_COLS = ["user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3"]
N_ITEMS = 80
N_POS = 3


@dataclass
class Logs:
    """Bandit feedback: one row per logged impression."""

    x: np.ndarray          # (n, d) one-hot user + position context
    affinity: np.ndarray   # (n, K) user-item affinity features
    action: np.ndarray     # (n,) logged item id
    position: np.ndarray   # (n,) 0-based slot
    reward: np.ndarray     # (n,) click
    pscore: np.ndarray     # (n,) logging propensity of the logged action
    timestamp: np.ndarray

    def __len__(self) -> int:
        return len(self.action)

    def subset(self, idx) -> "Logs":
        return Logs(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


def load_obd(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, nrows=nrows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


def featurize(df: pd.DataFrame, categories: dict | None = None) -> tuple[Logs, dict]:
    """One-hot user features (with fixed categories so train/test align) + position."""
    if categories is None:
        categories = {c: sorted(df[c].astype(str).unique()) for c in USER_COLS}
    blocks = []
    for c in USER_COLS:
        cat = pd.Categorical(df[c].astype(str), categories=categories[c])
        blocks.append(np.eye(len(categories[c]) + 1, dtype=np.float32)[np.where(cat.codes < 0, len(categories[c]), cat.codes)][:, :-1])
    pos = df["position"].to_numpy() - 1
    blocks.append(np.eye(N_POS, dtype=np.float32)[pos])
    x = np.hstack(blocks)
    aff_cols = [f"user-item_affinity_{k}" for k in range(N_ITEMS)]
    logs = Logs(
        x=x,
        affinity=df[aff_cols].to_numpy(np.float32),
        action=df["item_id"].to_numpy(),
        position=pos,
        reward=df["click"].to_numpy(np.float64),
        pscore=df["propensity_score"].to_numpy(np.float64),
        timestamp=df["timestamp"].to_numpy(),
    )
    return logs, categories


def time_split(logs: Logs, train_frac: float = 0.7) -> tuple[Logs, Logs]:
    """Chronological split: no future information leaks into training."""
    cut = int(len(logs) * train_frac)
    return logs.subset(slice(0, cut)), logs.subset(slice(cut, None))
