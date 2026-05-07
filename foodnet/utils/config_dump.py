"""Dump argparse.Namespace as args.json next to a checkpoint dir."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def serialize_args(args: argparse.Namespace) -> dict:
    """Return a JSON-safe dict view of an argparse Namespace (Paths -> str)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def dump_args(args: argparse.Namespace, out_dir: Path, filename: str = "args.json") -> Path:
    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(serialize_args(args), indent=2, sort_keys=True))
    return out
