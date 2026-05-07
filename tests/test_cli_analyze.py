import json
from pathlib import Path
import numpy as np
import pandas as pd
from foodnet.cli.analyze import main as analyze_main


def _make_preds(path: Path, n_per_class: int = 30, n_classes: int = 3) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_classes):
        for i in range(n_per_class):
            probs = rng.dirichlet(np.ones(n_classes))
            probs[c] += 0.6
            probs = probs / probs.sum()
            row = {"filepath": f"class_{c}/img_{i}.jpg", "label": c, "loss": float(-np.log(max(probs[c], 1e-12)))}
            for k in range(n_classes):
                row[f"p_{k}"] = float(probs[k])
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_analyze_cli_writes_dashboard(tmp_path: Path):
    parquet = tmp_path / "preds.parquet"
    _make_preds(parquet)
    classes_txt = tmp_path / "classes.txt"
    classes_txt.write_text("class_0\nclass_1\nclass_2\n")
    out = tmp_path / "dash"
    rc = analyze_main([
        "--predictions", str(parquet),
        "--classes", str(classes_txt),
        "--output-dir", str(out),
    ])
    assert rc == 0
    metrics = json.loads((out / "metrics.json").read_text())
    for key in ("top1", "top5", "macro_f1", "ece", "per_class_acc"):
        assert key in metrics
    assert (out / "confusion.png").exists()
    assert (out / "reliability.png").exists()
    assert (out / "dashboard.md").exists()
