import numpy as np
from foodnet.eval.confusion import confusion_matrix, most_confused_pairs


def test_confusion_matrix_shape_and_diagonal():
    softmax = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.9, 0.1, 0.0],
    ])
    targets = np.array([0, 1, 2, 2])
    cm = confusion_matrix(softmax, targets, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm[2, 0] == 1


def test_most_confused_pairs_excludes_diagonal_and_returns_top_k():
    cm = np.array([
        [10, 1, 0],
        [2, 8, 0],
        [3, 0, 7],
    ])
    pairs = most_confused_pairs(cm, k=2, class_names=["a", "b", "c"])
    assert len(pairs) == 2
    assert pairs[0]["count"] == 3
    assert pairs[0]["true"] == "c"
    assert pairs[0]["pred"] == "a"
