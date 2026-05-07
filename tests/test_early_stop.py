from foodnet.training.early_stop import EarlyStopper


def test_early_stop_max_mode_improves_resets_counter():
    es = EarlyStopper(patience=3, mode="max", min_delta=1e-3)
    assert not es.step(0.50)
    assert not es.step(0.55)
    assert not es.step(0.60)
    assert not es.step(0.55)  # 1
    assert not es.step(0.59)  # 2
    assert not es.step(0.6005)  # tiny improvement < min_delta -> 3 -> stop
    assert es.should_stop


def test_early_stop_returns_true_at_patience():
    es = EarlyStopper(patience=2, mode="max", min_delta=0.0)
    es.step(0.5)
    es.step(0.4)  # 1
    stop = es.step(0.3)  # 2 -> stop
    assert stop is True
    assert es.should_stop


def test_early_stop_min_mode():
    es = EarlyStopper(patience=2, mode="min", min_delta=1e-3)
    assert not es.step(1.0)
    assert not es.step(0.5)  # better
    assert not es.step(0.49)  # 1 (improvement < min_delta)
    assert es.step(0.50)  # 2 -> stop
