from foodnet.data.splits import load_split, load_class_names


def test_load_split_returns_filepaths_and_labels(tiny_food):
    paths, labels = load_split(tiny_food["splits_dir"] / "train.csv")
    assert len(paths) == len(labels)
    assert all(isinstance(p, str) for p in paths)
    assert set(labels) <= {0, 1, 2}


def test_load_class_names(tiny_food):
    names = load_class_names(tiny_food["splits_dir"] / "classes.txt")
    assert names == tiny_food["classes"]
