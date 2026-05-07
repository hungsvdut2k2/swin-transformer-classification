import torch
from torchvision import transforms as T
from foodnet.data.dataset import SplitsDataset


def _to_tensor():
    return T.Compose([T.Resize((16, 16)), T.ToTensor()])


def test_dataset_len_and_getitem(tiny_food):
    ds = SplitsDataset(
        csv_path=tiny_food["splits_dir"] / "train.csv",
        images_root=tiny_food["images_root"],
        transform=_to_tensor(),
    )
    assert len(ds) == 24
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 16, 16)
    assert isinstance(y, int) and 0 <= y <= 2


def test_dataset_resamples_on_pil_error(tiny_food):
    ds = SplitsDataset(
        csv_path=tiny_food["splits_dir"] / "train.csv",
        images_root=tiny_food["images_root"],
        transform=_to_tensor(),
    )
    bad = tiny_food["bad_file"]
    relpaths = [r for r, _ in zip(ds.filepaths, ds.labels)]
    if bad in relpaths:
        i = relpaths.index(bad)
        x, y = ds[i]
        assert isinstance(x, torch.Tensor)
