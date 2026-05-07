"""Dump argparse.Namespace as args.json next to a checkpoint dir."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def dump_args(args: argparse.Namespace, out_dir: Path, filename: str = "args.json") -> Path:
    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out
