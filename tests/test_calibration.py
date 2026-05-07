import numpy as np
from foodnet.eval.calibration import expected_calibration_error, reliability_bins


def test_ece_perfectly_calibrated_is_zero():
    # Confidence == accuracy in every bin → ECE = 0.
    n = 1000
    rng = np.random.default_rng(0)
    confs = rng.uniform(0.5, 1.0, size=n)
    correct = (rng.uniform(size=n) < confs).astype(np.int64)
    softmax = np.zeros((n, 2))
    softmax[np.arange(n), 0] = confs
    softmax[np.arange(n), 1] = 1 - confs
    targets = np.where(correct == 1, 0, 1)
    ece = expected_calibration_error(softmax, targets, n_bins=15)
    assert ece < 0.05


def test_reliability_bins_returns_15_bins():
    n = 100
    softmax = np.random.RandomState(0).rand(n, 5)
    softmax /= softmax.sum(axis=1, keepdims=True)
    targets = np.random.RandomState(1).randint(0, 5, size=n)
    bins = reliability_bins(softmax, targets, n_bins=15)
    assert len(bins["bin_lower"]) == 15
    assert len(bins["bin_upper"]) == 15
    assert len(bins["bin_acc"]) == 15
    assert len(bins["bin_conf"]) == 15
    assert len(bins["bin_count"]) == 15
