"""`foodnet-evaluate`: load checkpoint, run inference on one split, write parquet."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence
from torch.utils.data import DataLoader
from foodnet.data.dataset import SplitsDataset
from foodnet.data.transforms import build_eval_transform
from foodnet.models.factory import build_model
from foodnet.training.checkpoint import load_checkpoint
from foodnet.eval.runner import run_inference


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--arch", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--num-classes", type=int, default=101)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=False)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    transform = build_eval_transform(img_size=args.img_size)
    ds = SplitsDataset(args.splits_dir / f"{args.split}.csv", args.images_root, transform=transform)
    pin = args.device.startswith("cuda")
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=pin)
    model = build_model(args.arch, num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    load_checkpoint(args.checkpoint, model, optimizer=None, scaler=None, map_location=args.device)
    run_inference(model, dl, num_classes=args.num_classes, device=args.device, filepaths=ds.filepaths, out_path=args.output_parquet, amp=args.amp)
    print(f"[evaluate] wrote {len(ds)} predictions to {args.output_parquet}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
