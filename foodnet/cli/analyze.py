"""`foodnet-analyze`: read prediction parquet, write metrics + figures + dashboard.md."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from foodnet.data.splits import load_class_names
from foodnet.eval.metrics import topk_accuracy, macro_f1, per_class_accuracy
from foodnet.eval.confusion import confusion_matrix, most_confused_pairs
from foodnet.eval.calibration import expected_calibration_error, reliability_bins
from foodnet.eval.slices import worst_k_classes, hardest_examples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-analyze")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--worst-k", type=int, default=10)
    p.add_argument("--hardest-k", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=15)
    return p.parse_args(argv)


def _load(parquet: Path, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    cols = [f"p_{c}" for c in range(n_classes)]
    softmax = df[cols].to_numpy(dtype=np.float64)
    targets = df["label"].to_numpy(dtype=np.int64)
    losses = df["loss"].to_numpy(dtype=np.float64)
    filepaths = df["filepath"].to_numpy()
    return softmax, targets, losses, filepaths


def _plot_confusion(cm: np.ndarray, class_names: list[str], out: Path) -> None:
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n / 8), max(7, n / 8)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Confusion matrix (row-normalized, {n} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_reliability(bins: dict, ece: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    centers = (np.array(bins["bin_lower"]) + np.array(bins["bin_upper"])) / 2
    ax.bar(centers, bins["bin_acc"], width=1 / len(centers), edgecolor="k", alpha=0.6, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _write_dashboard(out: Path, metrics: dict, worst: list[dict], pairs: list[dict], hardest_correct: list[dict], hardest_incorrect: list[dict]) -> None:
    md = ["# Evaluation Dashboard\n"]
    md.append("## Headline metrics\n")
    md.append(f"- Top-1: {metrics['top1']:.4f}")
    md.append(f"- Top-5: {metrics['top5']:.4f}")
    md.append(f"- Macro-F1: {metrics['macro_f1']:.4f}")
    md.append(f"- ECE: {metrics['ece']:.4f}\n")
    md.append("## Worst-K classes\n")
    for r in worst:
        md.append(f"- {r['class']}: {r['accuracy']:.3f}")
    md.append("\n## Most-confused pairs (true → pred, count)\n")
    for r in pairs:
        md.append(f"- {r['true']} → {r['pred']}: {r['count']}")
    md.append("\n## Hardest correct examples\n")
    for r in hardest_correct:
        md.append(f"- idx={r['index']} loss={r['loss']:.3f} target={r['target']} pred={r['pred']}")
    md.append("\n## Hardest incorrect examples\n")
    for r in hardest_incorrect:
        md.append(f"- idx={r['index']} loss={r['loss']:.3f} target={r['target']} pred={r['pred']}")
    md.append("\n## Figures\n")
    md.append("- ![confusion](confusion.png)")
    md.append("- ![reliability](reliability.png)\n")
    out.write_text("\n".join(md))


def run(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.classes)
    n_classes = len(class_names)
    softmax, targets, losses, _filepaths = _load(args.predictions, n_classes)

    top1 = topk_accuracy(softmax, targets, k=1)
    top5 = topk_accuracy(softmax, targets, k=5)
    f1 = macro_f1(softmax, targets, num_classes=n_classes)
    pca = per_class_accuracy(softmax, targets, num_classes=n_classes)
    cm = confusion_matrix(softmax, targets, num_classes=n_classes)
    pairs = most_confused_pairs(cm, k=20, class_names=class_names)
    bins = reliability_bins(softmax, targets, n_bins=args.n_bins)
    ece = expected_calibration_error(softmax, targets, n_bins=args.n_bins)
    worst = worst_k_classes(pca, k=args.worst_k, class_names=class_names)
    preds = softmax.argmax(axis=1)
    hardest_correct = hardest_examples(losses, preds, targets, k=args.hardest_k, kind="correct")
    hardest_incorrect = hardest_examples(losses, preds, targets, k=args.hardest_k, kind="incorrect")

    metrics = {
        "top1": top1, "top5": top5, "macro_f1": f1, "ece": ece,
        "per_class_acc": {class_names[i]: float(pca[i]) for i in range(n_classes)},
        "n": int(len(targets)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion.csv")
    (out_dir / "most_confused.json").write_text(json.dumps(pairs, indent=2))
    (out_dir / "reliability.json").write_text(json.dumps({"ece": ece, **bins}, indent=2))
    _plot_confusion(cm, class_names, out_dir / "confusion.png")
    _plot_reliability(bins, ece, out_dir / "reliability.png")
    _write_dashboard(out_dir / "dashboard.md", metrics, worst, pairs[:20], hardest_correct, hardest_incorrect)
    print(f"[analyze] wrote dashboard to {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
