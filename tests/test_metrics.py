import numpy as np
from foodnet.eval.metrics import topk_accuracy, macro_f1, per_class_accuracy


def test_topk_accuracy_perfect():
    softmax = np.eye(5)
    targets = np.arange(5)
    assert topk_accuracy(softmax, targets, k=1) == 1.0
    assert topk_accuracy(softmax, targets, k=5) == 1.0


def test_topk_accuracy_top5_includes_top1():
    softmax = np.array([
        [0.1, 0.2, 0.0, 0.6, 0.1],  # true=3 -> top1
        [0.0, 0.0, 0.4, 0.5, 0.1],  # true=1 -> not in top1, in top5 (rank 5)
    ])
    targets = np.array([3, 1])
    assert topk_accuracy(softmax, targets, k=1) == 0.5
    assert topk_accuracy(softmax, targets, k=5) == 1.0


def test_macro_f1_balanced_dataset():
    softmax = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    targets = np.array([0, 1, 0, 1])
    f1 = macro_f1(softmax, targets, num_classes=2)
    assert f1 == 1.0


def test_per_class_accuracy_returns_one_per_class():
    softmax = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.9, 0.1, 0.0],  # wrong (true=2)
    ])
    targets = np.array([0, 1, 2, 2])
    pca = per_class_accuracy(softmax, targets, num_classes=3)
    assert len(pca) == 3
    assert pca[0] == 1.0
    assert pca[1] == 1.0
    assert pca[2] == 0.5
