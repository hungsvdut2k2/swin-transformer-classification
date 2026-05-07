import numpy as np
from foodnet.eval.slices import worst_k_classes, hardest_examples


def test_worst_k_classes_returns_lowest_acc():
    pca = np.array([0.9, 0.5, 0.7, 0.2])
    out = worst_k_classes(pca, k=2, class_names=["a", "b", "c", "d"])
    assert [r["class"] for r in out] == ["d", "b"]
    assert out[0]["accuracy"] == 0.2


def test_hardest_correct_high_loss_correct_only():
    losses = np.array([0.1, 2.0, 0.5, 3.0, 0.05])
    preds = np.array([0, 0, 0, 0, 0])
    targets = np.array([0, 1, 0, 1, 0])  # correct: 0,2,4
    out = hardest_examples(losses, preds, targets, k=2, kind="correct")
    assert len(out) == 2
    assert set(o["index"] for o in out) == {2, 0}


def test_hardest_incorrect_high_loss_incorrect_only():
    losses = np.array([0.1, 2.0, 0.5, 3.0, 0.05])
    preds = np.array([0, 0, 0, 0, 0])
    targets = np.array([0, 1, 0, 1, 0])  # incorrect: 1,3
    out = hardest_examples(losses, preds, targets, k=2, kind="incorrect")
    assert [o["index"] for o in out] == [3, 1]
