"""`foodnet-split` CLI: scan an images_root and write 8:1:1 split CSVs."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence
import pandas as pd
from tqdm.auto import tqdm
from foodnet.splitting.stratified import stratified_split


IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-split", description="Generate 8:1:1 stratified splits over an images_root.")
    p.add_argument("--images-root", type=Path, required=True, help="Directory whose immediate children are class folders.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write {train,val,test}.csv and metadata.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="Comma-separated train,val,test ratios.")
    return p.parse_args(argv)


def _scan(images_root: Path) -> tuple[pd.DataFrame, list[str]]:
    classes = sorted(d.name for d in images_root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class subdirectories found under {images_root}.")
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    rows: list[dict] = []
    for c in tqdm(classes, desc="scan", leave=False, dynamic_ncols=True):
        for f in sorted((images_root / c).iterdir()):
            if f.suffix.lower() in IMG_EXT and f.is_file():
                rows.append({"filepath": f"{c}/{f.name}", "label": cls_to_idx[c], "class_name": c})
    return pd.DataFrame(rows), classes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args: argparse.Namespace) -> int:
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit(f"--ratios must be 3 comma-separated floats, got {args.ratios!r}")
    df, classes = _scan(args.images_root)
    train, val, test = stratified_split(df, ratios=ratios, seed=args.seed, progress=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "classes.txt").write_text("\n".join(classes) + "\n")
    meta = {
        "seed": args.seed,
        "ratios": list(ratios),
        "n_classes": len(classes),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "sha256": {f"{n}.csv": _sha256(args.output_dir / f"{n}.csv") for n in ("train", "val", "test")},
    }
    (args.output_dir / "splits_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[split] wrote {len(train)}/{len(val)}/{len(test)} rows to {args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
