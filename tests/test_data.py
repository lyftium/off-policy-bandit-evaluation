import pandas as pd
import numpy as np

from cbope.data import featurize, time_split, USER_COLS


def _df(n=50):
    rng = np.random.default_rng(0)
    d = {c: rng.choice(["a", "b", "c"], n) for c in USER_COLS}
    d.update(timestamp=pd.date_range("2019-11-24", periods=n, freq="min"),
             item_id=rng.integers(0, 80, n), position=rng.integers(1, 4, n),
             click=rng.integers(0, 2, n), propensity_score=np.full(n, 1 / 80))
    for k in range(80):
        d[f"user-item_affinity_{k}"] = 0.0
    return pd.DataFrame(d)


def test_featurize_shapes_and_unseen_category():
    logs, cats = featurize(_df())
    assert logs.x.shape == (50, 12 + 3)
    df2 = _df(5); df2.loc[0, "user_feature_0"] = "zzz"
    logs2, _ = featurize(df2, cats)
    assert logs2.x[0, :3].sum() == 0  # unseen -> all zeros, no crash


def test_time_split_is_chronological():
    logs, _ = featurize(_df())
    tr, te = time_split(logs, 0.7)
    assert len(tr) == 35 and tr.timestamp.max() < te.timestamp.min()
