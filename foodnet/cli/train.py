"""`foodnet-train`: end-to-end training CLI for the Swin-Tiny baseline."""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy as _timm_acc  # noqa: F401  (kept for symmetry; we use our own loop)
from foodnet.data.dataset import SplitsDataset
from foodnet.data.transforms import build_train_transform, build_eval_transform
from foodnet.data.mixup import build_mixup_fn
from foodnet.models.factory import build_model
from foodnet.models.llrd import llrd_param_groups
from foodnet.training.optim import build_optimizer, build_scheduler, build_scaler
from foodnet.training.loop import train_one_epoch, validate
from foodnet.training.checkpoint import save_checkpoint, load_checkpoint
from foodnet.training.early_stop import EarlyStopper
from foodnet.utils.seed import set_seed
from foodnet.utils.paths import ensure_dir
from foodnet.utils.wandb_logger import WandBLogger
from foodnet.utils.config_dump import dump_args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-train")

    g = p.add_argument_group("data")
    g.add_argument("--splits-dir", type=Path, required=True)
    g.add_argument("--images-root", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--num-classes", type=int, default=101)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--batch-size", type=int, default=64)

    g = p.add_argument_group("model")
    g.add_argument("--arch", type=str, default="swin_tiny_patch4_window7_224")
    g.add_argument("--drop-path", type=float, default=0.2)
    g.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    g.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--layer-decay", type=float, default=0.75)
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--warmup-epochs", type=int, default=2)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--label-smoothing", type=float, default=0.1)

    g = p.add_argument_group("aug")
    g.add_argument("--mixup-alpha", type=float, default=0.8)
    g.add_argument("--cutmix-alpha", type=float, default=1.0)
    g.add_argument("--re-prob", type=float, default=0.25)
    g.add_argument("--color-jitter", type=float, default=0.4)
    g.add_argument("--auto-augment", type=str, default="rand-m9-mstd0.5-inc1")

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--amp", dest="amp", action="store_true", default=True)
    g.add_argument("--no-amp", dest="amp", action="store_false")
    g.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    g.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    g.add_argument("--early-stop-patience", type=int, default=8)
    g.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    g.add_argument("--resume", action="store_true", default=False)

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    g.add_argument("--no-wandb", dest="wandb", action="store_false")
    g.add_argument("--wandb-project", type=str, default="foodnet")
    g.add_argument("--run-name", type=str, default=None)

    return p.parse_args(argv)


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    t_train = build_train_transform(
        img_size=args.img_size,
        auto_augment=args.auto_augment,
        re_prob=args.re_prob,
        color_jitter=args.color_jitter,
    )
    t_eval = build_eval_transform(img_size=args.img_size)
    train_ds = SplitsDataset(args.splits_dir / "train.csv", args.images_root, transform=t_train)
    val_ds = SplitsDataset(args.splits_dir / "val.csv", args.images_root, transform=t_eval)
    pin = args.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    return train_dl, val_dl


def run(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.output_dir)
    last_pt = out_dir / "last.pt"
    if last_pt.exists() and not args.resume:
        raise SystemExit(f"Refusing to clobber {last_pt}. Pass --resume to continue or pick a fresh --output-dir.")

    set_seed(args.seed)
    dump_args(args, out_dir)

    train_dl, val_dl = _build_loaders(args)
    device = args.device
    model = build_model(args.arch, num_classes=args.num_classes, pretrained=args.pretrained, drop_path_rate=args.drop_path).to(device)
    mixup_fn = build_mixup_fn(args.num_classes, args.mixup_alpha, args.cutmix_alpha, args.label_smoothing)

    if mixup_fn is not None:
        train_criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    groups = llrd_param_groups(model, base_lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay)
    optimizer = build_optimizer(groups, kind="adamw")
    scheduler = build_scheduler(optimizer, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    scaler = build_scaler(enabled=args.amp and device.startswith("cuda"))

    start_epoch, best_top1 = 0, 0.0
    if args.resume and last_pt.exists():
        state = load_checkpoint(last_pt, model, optimizer, scaler, map_location=device)
        start_epoch = state["epoch"] + 1
        best_top1 = state["best_metric"]
        tqdm.write(f"[train] resumed from epoch {start_epoch} (best_top1={best_top1:.4f})")

    stopper = EarlyStopper(patience=args.early_stop_patience, mode="max", min_delta=args.early_stop_min_delta) if args.early_stop else None
    logger = WandBLogger(enabled=args.wandb, project=args.wandb_project, run_name=args.run_name, config=vars(args))
    log_csv = out_dir / "train_log.csv"
    new_log = not log_csv.exists()

    with log_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "elapsed_s"])
        epoch_bar = tqdm(
            range(start_epoch, args.epochs),
            desc="epochs",
            initial=start_epoch,
            total=args.epochs,
            dynamic_ncols=True,
        )
        for epoch in epoch_bar:
            t0 = time.time()
            tr = train_one_epoch(
                model, train_dl, optimizer, train_criterion, scaler,
                device=device, grad_clip=args.grad_clip, mixup_fn=mixup_fn, amp=args.amp,
                progress=True, desc=f"train ep {epoch}",
            )
            scheduler.step(epoch + 1)
            va = validate(
                model, val_dl, val_criterion,
                device=device, num_classes=args.num_classes, amp=args.amp,
                progress=True, desc=f"val ep {epoch}",
            )
            elapsed = time.time() - t0
            writer.writerow([epoch, f"{tr['loss']:.6f}", f"{va['loss']:.6f}", f"{va['top1']:.6f}", f"{va['top5']:.6f}", f"{elapsed:.2f}"])
            fh.flush()
            logger.log({"train/loss": tr["loss"], "val/loss": va["loss"], "val/top1": va["top1"], "val/top5": va["top5"], "epoch": epoch}, step=epoch)
            tqdm.write(f"[train] epoch={epoch} train_loss={tr['loss']:.4f} val_top1={va['top1']:.4f} val_top5={va['top5']:.4f} ({elapsed:.1f}s)")
            epoch_bar.set_postfix(top1=f"{va['top1']:.4f}", best=f"{max(best_top1, va['top1']):.4f}")

            save_checkpoint(last_pt, model, optimizer, scaler, epoch=epoch, best_metric=best_top1)
            if va["top1"] > best_top1:
                best_top1 = va["top1"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, epoch=epoch, best_metric=best_top1)
            if stopper is not None and stopper.step(va["top1"]):
                tqdm.write(f"[train] early stop at epoch {epoch} (best={stopper.best:.4f})")
                break
        epoch_bar.close()

    logger.finish()
    print(f"[train] done. best val/top1 = {best_top1:.4f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
