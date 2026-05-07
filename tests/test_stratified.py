from pathlib import Path
import pandas as pd
import pytest
from foodnet.splitting.stratified import stratified_split


def _make_df(n_per_class: int = 1000, n_classes: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_classes):
        for i in range(n_per_class):
            rows.append({"filepath": f"class_{c}/img_{i:04d}.jpg", "label": c, "class_name": f"class_{c}"})
    return pd.DataFrame(rows)


def test_stratified_split_sizes_per_class():
    df = _make_df(1000, 3)
    train, val, test = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    for cls in range(3):
        assert (train["label"] == cls).sum() == 800
        assert (val["label"] == cls).sum() == 100
        assert (test["label"] == cls).sum() == 100


def test_stratified_split_disjoint():
    df = _make_df(1000, 3)
    train, val, test = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    s_train = set(train["filepath"])
    s_val = set(val["filepath"])
    s_test = set(test["filepath"])
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train) + len(s_val) + len(s_test) == 3000


def test_stratified_split_deterministic():
    df = _make_df(100, 3)
    a = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    b = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    for da, db in zip(a, b):
        pd.testing.assert_frame_equal(da.reset_index(drop=True), db.reset_index(drop=True))


def test_stratified_split_ratios_must_sum_to_one():
    df = _make_df(10, 2)
    with pytest.raises(ValueError):
        stratified_split(df, ratios=(0.7, 0.1, 0.1), seed=42)
