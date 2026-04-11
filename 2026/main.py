#!/usr/bin/env python3
"""Backdoor attack + adversarial training pipeline.

Supports any torchvision image-classification dataset (MNIST, CIFAR-10, ...).

Usage examples
--------------
# MNIST, single run (default):
  python main.py

# CIFAR-10, poison 15%, adversarial training, target label 3:
  python main.py --dataset cifar10 --alpha 0.15 --adv-train --target 3

# CIFAR-10 with a pretrained robust model from RobustBench (no training):
  python main.py --dataset cifar10 --alpha 0.15 --robustbench Carmon2019Unlabeled

# Full sweep across alphas using a RobustBench model:
  python main.py --dataset cifar10 --sweep --target 7 --robustbench Wong2020Fast

# Disable TensorBoard logging:
  python main.py --no-tensorboard
"""

from __future__ import annotations

import warnings

# torchvision's CIFAR loader passes align=0 (int) to numpy.dtype(); NumPy ≥2.4
# expects a bool.  Harmless, but noisy — suppress until torchvision ships a fix.
warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean",
    category=DeprecationWarning,
)

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from attacks import pgd
from backdoor import BackdoorConfig, BackdoorStyle, poison_dataset, stamp
from model import SmallCNN, compile_model, _get_amp_dtype, train


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about a supported dataset."""
    in_channels: int
    img_size: int
    num_classes: int
    default_trigger_pos: tuple[int, int]  # sensible default near bottom-right
    robustbench_dataset: str | None  # name used by robustbench.utils.load_model


DATASETS: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo(in_channels=1, img_size=28, num_classes=10, default_trigger_pos=(25, 25), robustbench_dataset=None),
    "fashion_mnist": DatasetInfo(in_channels=1, img_size=28, num_classes=10, default_trigger_pos=(25, 25), robustbench_dataset=None),
    "cifar10": DatasetInfo(in_channels=3, img_size=32, num_classes=10, default_trigger_pos=(29, 29), robustbench_dataset="cifar10"),
    "cifar100": DatasetInfo(in_channels=3, img_size=32, num_classes=100, default_trigger_pos=(29, 29), robustbench_dataset="cifar100"),
}

_TORCHVISION_CLS: dict[str, type] = {
    "mnist": torchvision.datasets.MNIST,
    "fashion_mnist": torchvision.datasets.FashionMNIST,
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}


def load_dataset(
    name: str, data_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DatasetInfo]:
    """Return (train_images, train_labels, test_images, test_labels, info).

    Images are float32 in [0, 1] with shape (N, C, H, W).
    Labels are int64.
    """
    info = DATASETS[name]
    cls = _TORCHVISION_CLS[name]

    train_ds = cls(data_dir, train=True, download=True)
    test_ds = cls(data_dir, train=False, download=True)

    def _to_tensors(ds: torchvision.datasets.VisionDataset) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(ds, "data") and isinstance(ds.data, torch.Tensor):
            # MNIST / FashionMNIST: ds.data is uint8 (N, H, W)
            imgs = ds.data.float() / 255.0
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(1)  # (N, 1, H, W)
            labels = ds.targets
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            return imgs, labels.long()
        else:
            # CIFAR: ds.data is numpy uint8 (N, H, W, C)
            import numpy as np
            imgs = torch.from_numpy(np.array(ds.data)).float() / 255.0
            if imgs.ndim == 4:
                imgs = imgs.permute(0, 3, 1, 2)  # (N, C, H, W)
            labels = torch.tensor(ds.targets).long()
            return imgs, labels

    train_images, train_labels = _to_tensors(train_ds)
    test_images, test_labels = _to_tensors(test_ds)
    return train_images, train_labels, test_images, test_labels, info


# ---------------------------------------------------------------------------
# RobustBench integration
# ---------------------------------------------------------------------------

def load_robustbench_model(
    model_name: str,
    dataset_info: DatasetInfo,
    threat_model: str = "Linf",
    model_dir: str = "./models",
) -> nn.Module:
    """Download and return a pretrained robust model from RobustBench.

    The returned model is a standard nn.Module in eval mode.  It expects
    float32 inputs in [0, 1] — normalization is applied internally.

    Supported datasets: cifar10, cifar100 (and imagenet, but not wired here).
    Popular CIFAR-10 Linf model names:
        Carmon2019Unlabeled, Wong2020Fast, Rice2020Overfitting,
        Engstrom2019Robustness, Rebuffi2021Fixing_70_16_cutmix_extra, ...
    Full list: https://robustbench.github.io/
    """
    if dataset_info.robustbench_dataset is None:
        raise ValueError(
            f"RobustBench does not have models for this dataset. "
            f"Supported: {[k for k, v in DATASETS.items() if v.robustbench_dataset]}"
        )
    from robustbench.utils import load_model
    return load_model(
        model_name=model_name,
        dataset=dataset_info.robustbench_dataset,
        threat_model=threat_model,
        model_dir=model_dir,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512, desc: str = "accuracy") -> float:
    model.eval()
    device = next(model.parameters()).device
    amp_dtype = _get_amp_dtype(device)
    use_amp = amp_dtype is not None
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    correct = total = 0
    for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


def robust_eval(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    pgd_alpha: float,
    pgd_iter: int,
    pgd_restarts: int,
    batch_size: int = 128,
    desc: str = "robust eval",
) -> tuple[float, float]:
    r"""Compute robust loss and robust error rate under PGD attack.

    Returns ``(robust_loss, robust_error_rate)`` where:

    - **robust_loss**: expected worst-case cross-entropy,
      :math:`\frac{1}{m}\sum_{i=1}^{m}\max_{\hat x \in B(x_i,\delta)} L_h(\hat x, y_i)`
    - **robust_error_rate**: fraction misclassified after PGD,
      :math:`1 - \text{robust\_accuracy}` (matches the original TF ``Robust Loss`` metric).
    """
    model.eval()
    device = next(model.parameters()).device
    amp_dtype = _get_amp_dtype(device)
    use_amp = amp_dtype is not None
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    total_loss = 0.0
    wrong = 0
    total = 0
    for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            x_adv = pgd(model, xb, yb, eps=eps, alpha=pgd_alpha,
                         num_iter=pgd_iter, restarts=pgd_restarts)
            with torch.no_grad():
                logits = model(x_adv)
                total_loss += loss_fn(logits, yb).sum().item()
                wrong += (logits.argmax(1) != yb).sum().item()
        total += len(xb)
    return total_loss / total, wrong / total


def backdoor_success_rate(
    model: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    cfg: BackdoorConfig,
    batch_size: int = 512,
) -> float:
    """Fraction of non-target test images classified as target after stamping."""
    mask = test_labels != cfg.target_label
    imgs = stamp(test_images[mask], cfg)
    labels = torch.full((imgs.size(0),), cfg.target_label, dtype=test_labels.dtype, device=imgs.device)
    return accuracy(model, imgs, labels, batch_size)


# ---------------------------------------------------------------------------
# TensorBoard image logging
# ---------------------------------------------------------------------------

def _log_image_grid(
    writer: SummaryWriter,
    images: torch.Tensor,
    labels: torch.Tensor,
    tag: str,
    global_step: int,
    n: int = 16,
) -> None:
    """Log a grid of images with a per-image label annotation in the tag."""
    imgs = images[:n].cpu()
    labs = labels[:n].cpu().tolist()
    grid = vutils.make_grid(imgs, nrow=8, normalize=False, padding=2)
    label_str = ",".join(str(l) for l in labs)
    writer.add_image(f"{tag} [labels={label_str}]", grid, global_step)


def log_sample_images(
    writer: SummaryWriter,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: BackdoorConfig,
    global_step: int,
    n: int = 16,
) -> None:
    """Log grids of clean and backdoored training images to TensorBoard."""
    # Clean examples — pick a diverse set (first n)
    _log_image_grid(writer, train_images, train_labels, "images/clean", global_step, n)

    if cfg.alpha > 0:
        # Pick non-target images, stamp them, show with both original and target labels
        mask = train_labels != cfg.target_label
        src_imgs = train_images[mask][:n].cpu()
        src_labs = train_labels[mask][:n].cpu()

        stamped = stamp(src_imgs, cfg)
        orig_str = ",".join(str(l.item()) for l in src_labs)
        target_str = str(cfg.target_label)

        grid = vutils.make_grid(stamped, nrow=8, normalize=False, padding=2)
        writer.add_image(
            f"images/backdoored [orig={orig_str} -> target={target_str}]",
            grid,
            global_step,
        )


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(
    *,
    dataset_name: str,
    target: int,
    alpha: float,
    adv_train: bool,
    style: BackdoorStyle,
    color: float,
    position: tuple[int, int] | None,
    source_label: int,
    backdoor_eps: float | None,
    epochs: int,
    batch_size: int,
    lr: float,
    pgd_eps: float,
    pgd_alpha: float,
    pgd_iter: int,
    pgd_restarts: int,
    eval_subsample: int,
    checkpoint: str | None,
    robustbench_model: str | None,
    robustbench_threat: str,
    device: torch.device,
    seed: int,
    writer: SummaryWriter | None = None,
    run_index: int = 0,
) -> dict:
    rng = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    # 1. Load & poison
    train_images, train_labels, test_images, test_labels, info = load_dataset(dataset_name)

    if position is None:
        position = info.default_trigger_pos

    cfg = BackdoorConfig(
        style=style, color=color, position=position,
        target_label=target, alpha=alpha, source_label=source_label,
        eps=backdoor_eps,
    )
    p_images, p_labels = poison_dataset(train_images, train_labels, cfg, batch_size=batch_size, rng=rng)

    # Log sample images before moving to device
    if writer is not None:
        log_sample_images(writer, train_images, train_labels, cfg, global_step=run_index)
        writer.flush()

    p_images, p_labels = p_images.to(device), p_labels.to(device)
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    # 2. Build / load model
    if robustbench_model:
        model = compile_model(load_robustbench_model(
            robustbench_model, info,
            threat_model=robustbench_threat,
        ).to(device))
        print(f"  loaded RobustBench model: {robustbench_model} ({robustbench_threat})")
    elif checkpoint:
        model = SmallCNN(
            in_channels=info.in_channels,
            img_size=info.img_size,
            num_classes=info.num_classes,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model = compile_model(model)
        print(f"  loaded checkpoint {checkpoint}")
    else:
        model = SmallCNN(
            in_channels=info.in_channels,
            img_size=info.img_size,
            num_classes=info.num_classes,
        ).to(device)
        model = train(
            model, p_images, p_labels,
            epochs=epochs, batch_size=batch_size, lr=lr,
            adv_train=adv_train,
            pgd_eps=pgd_eps, pgd_alpha=pgd_alpha,
            pgd_iter=pgd_iter, pgd_restarts=pgd_restarts,
            device=device,
            writer=writer,
            global_step_offset=run_index * epochs,
        )

    # 3. Evaluate
    idx = torch.randperm(len(p_images), generator=rng)[:eval_subsample]
    sub_images, sub_labels = p_images[idx], p_labels[idx]

    print("  evaluating...")
    train_acc = accuracy(model, sub_images, sub_labels, desc="train accuracy")
    train_rloss, train_rerr = robust_eval(model, sub_images, sub_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts, desc="train robust")
    test_acc = accuracy(model, test_images, test_labels, desc="test accuracy")
    test_rloss, test_rerr = robust_eval(model, test_images, test_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts, desc="test robust")
    bd_rate = backdoor_success_rate(model, test_images, test_labels, cfg) if alpha > 0 else 0.0

    results = {
        "dataset": dataset_name,
        "alpha": alpha,
        "adv_train": adv_train,
        "robustbench_model": robustbench_model,
        "target": target,
        "train": {"accuracy": train_acc, "binary_loss": 1 - train_acc, "robust_loss": train_rloss, "robust_error": train_rerr},
        "test": {"accuracy": test_acc, "binary_loss": 1 - test_acc, "robust_loss": test_rloss, "robust_error": test_rerr},
        "backdoor_success": bd_rate,
    }

    # Log eval scalars
    if writer is not None:
        step = run_index
        writer.add_scalar("eval/train_accuracy", train_acc, step)
        writer.add_scalar("eval/train_binary_loss", 1 - train_acc, step)
        writer.add_scalar("eval/train_robust_loss", train_rloss, step)
        writer.add_scalar("eval/train_robust_error", train_rerr, step)
        writer.add_scalar("eval/test_accuracy", test_acc, step)
        writer.add_scalar("eval/test_binary_loss", 1 - test_acc, step)
        writer.add_scalar("eval/test_robust_loss", test_rloss, step)
        writer.add_scalar("eval/test_robust_error", test_rerr, step)
        if alpha > 0:
            writer.add_scalar("eval/backdoor_success", bd_rate, step)
        writer.flush()

    tag = robustbench_model or ("adv" if adv_train else "std")
    print(
        f"  [{tag}] alpha={alpha:.2f}  "
        f"train_acc={train_acc:.3f}  train_robust_loss={train_rloss:.4f}  train_robust_err={train_rerr:.3f}  "
        f"test_acc={test_acc:.3f}  test_robust_loss={test_rloss:.4f}  test_robust_err={test_rerr:.3f}  "
        f"backdoor={bd_rate:.3f}"
    )

    if alpha > 0:
        print(f"  -> Training set is backdoored (alpha={alpha:.2f}).")
    else:
        print("  -> Clean training set.")

    return results


def run_pretrained_eval(
    *,
    dataset_name: str,
    target: int,
    alpha: float,
    style: BackdoorStyle,
    color: float,
    position: tuple[int, int] | None,
    source_label: int,
    backdoor_eps: float | None,
    pgd_eps: float,
    pgd_alpha: float,
    pgd_iter: int,
    pgd_restarts: int,
    eval_subsample: int,
    robustbench_model: str,
    robustbench_threat: str,
    device: torch.device,
    seed: int,
    writer: SummaryWriter | None = None,
    run_index: int = 0,
) -> dict:
    """Eval-only path for pretrained robust models.

    Poisons the training set, then reports accuracy and robust loss on both
    the backdoored training set and the clean validation set.  No training.
    """
    rng = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    # 1. Load & poison
    train_images, train_labels, test_images, test_labels, info = load_dataset(dataset_name)

    if position is None:
        position = info.default_trigger_pos

    cfg = BackdoorConfig(
        style=style, color=color, position=position,
        target_label=target, alpha=alpha, source_label=source_label,
        eps=backdoor_eps,
    )
    p_images, p_labels = poison_dataset(train_images, train_labels, cfg, rng=rng)

    if writer is not None:
        log_sample_images(writer, train_images, train_labels, cfg, global_step=run_index)
        writer.flush()

    p_images, p_labels = p_images.to(device), p_labels.to(device)
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    # 2. Load pretrained model
    model = compile_model(load_robustbench_model(
        robustbench_model, info, threat_model=robustbench_threat,
    ).to(device))
    print(f"  loaded RobustBench model: {robustbench_model} ({robustbench_threat})")

    # 3. Evaluate — accuracy + robust error on train and val
    idx = torch.randperm(len(p_images), generator=rng)[:eval_subsample]
    sub_images, sub_labels = p_images[idx], p_labels[idx]

    print("  evaluating...")
    train_acc = accuracy(model, sub_images, sub_labels, desc="train accuracy")
    train_rloss, train_rerr = robust_eval(model, sub_images, sub_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts, desc="train robust")
    val_acc = accuracy(model, test_images, test_labels, desc="val accuracy")
    val_rloss, val_rerr = robust_eval(model, test_images, test_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts, desc="val robust")
    bd_rate = backdoor_success_rate(model, test_images, test_labels, cfg) if alpha > 0 else 0.0

    results = {
        "dataset": dataset_name,
        "alpha": alpha,
        "robustbench_model": robustbench_model,
        "robustbench_threat": robustbench_threat,
        "target": target,
        "train": {"accuracy": train_acc, "binary_loss": 1 - train_acc, "robust_loss": train_rloss, "robust_error": train_rerr},
        "val": {"accuracy": val_acc, "binary_loss": 1 - val_acc, "robust_loss": val_rloss, "robust_error": val_rerr},
        "backdoor_success": bd_rate,
    }

    if writer is not None:
        step = run_index
        writer.add_scalar("eval/train_accuracy", train_acc, step)
        writer.add_scalar("eval/train_robust_loss", train_rloss, step)
        writer.add_scalar("eval/train_robust_error", train_rerr, step)
        writer.add_scalar("eval/val_accuracy", val_acc, step)
        writer.add_scalar("eval/val_robust_loss", val_rloss, step)
        writer.add_scalar("eval/val_robust_error", val_rerr, step)
        if alpha > 0:
            writer.add_scalar("eval/backdoor_success", bd_rate, step)
        writer.flush()

    print(
        f"  [{robustbench_model}] alpha={alpha:.2f}  "
        f"train_acc={train_acc:.3f}  train_robust_loss={train_rloss:.4f}  train_robust_err={train_rerr:.3f}  "
        f"val_acc={val_acc:.3f}  val_robust_loss={val_rloss:.4f}  val_robust_err={val_rerr:.3f}  "
        f"backdoor={bd_rate:.3f}"
    )

    if alpha > 0:
        print(f"  -> Training set is backdoored (alpha={alpha:.2f}).")
    else:
        print("  -> Clean training/val sets.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backdoor + adversarial-training pipeline (PyTorch)")

    # Experiment mode
    p.add_argument("--sweep", action="store_true", help="Run full alpha x adv_train sweep")

    # Dataset
    p.add_argument("--dataset", choices=list(DATASETS), default="mnist", help="Dataset (default: mnist)")

    # Backdoor
    p.add_argument("--target", type=int, default=0, help="Target label for backdoor (default: 0)")
    p.add_argument("--alpha", type=float, default=0.0, help="Fraction of dataset to poison (default: 0.0)")
    p.add_argument("--style", choices=["pixel", "pattern", "ell"], default="pattern", help="Backdoor trigger style")
    p.add_argument("--color", type=float, default=0.3, help="Trigger pixel intensity 0-1 (default: 0.3)")
    p.add_argument("--position", type=int, nargs=2, default=None, help="Trigger position row col (default: dataset-dependent)")
    p.add_argument("--source-label", type=int, default=-1, help="Source class to poison (-1 = all non-target)")
    p.add_argument(
        "--backdoor-eps", type=float, default=None, metavar="EPS",
        help="If set, clamp backdoor trigger perturbation to Linf <= EPS "
             "(e.g. 0.03137 for 8/255). Without this, trigger pixels are "
             "overwritten to --color directly (unbounded).",
    )

    # Model source
    model_src = p.add_argument_group("model source")
    model_src.add_argument("--adv-train", action="store_true", help="Enable adversarial training (from scratch)")
    model_src.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained model .pt file")
    model_src.add_argument(
        "--robustbench", type=str, default=None, metavar="MODEL",
        help="Load a pretrained robust model from RobustBench by name "
             "(e.g. Carmon2019Unlabeled, Wong2020Fast). "
             "Skips training entirely — eval only. "
             "Requires: pip install robustbench",
    )
    model_src.add_argument(
        "--robustbench-threat", type=str, default="Linf",
        choices=["Linf", "L2", "corruptions"],
        help="RobustBench threat model (default: Linf)",
    )

    # Training (ignored when --robustbench is set)
    p.add_argument("--epochs", type=int, default=2, help="Training epochs (default: 2)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")

    # PGD
    p.add_argument("--pgd-eps", type=float, default=0.3, help="PGD epsilon (default: 0.3)")
    p.add_argument("--pgd-alpha", type=float, default=0.01, help="PGD step size (default: 0.01)")
    p.add_argument("--pgd-iter", type=int, default=40, help="PGD iterations (default: 40)")
    p.add_argument("--pgd-restarts", type=int, default=10, help="PGD random restarts (default: 10)")

    # Eval
    p.add_argument("--eval-subsample", type=int, default=5000, help="Training subset size for robustness eval")

    # TensorBoard
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    p.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory (default: runs)")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    p.add_argument("--results-dir", type=str, default="results", help="Directory for result JSON files")

    return p.parse_args(argv)


def _make_run_dir(
    logdir: str,
    args: argparse.Namespace,
    adv: bool | None = None,
    alpha: float | None = None,
) -> Path:
    """Create and return a unique, timestamped run directory.

    Layout: <logdir>/<dataset>_target<T>_alpha<A>_<mode>_<style>_<YYYYMMDD_HHMMSS>/
    A ``hparams.json`` file is written inside with all hyperparameters.
    """
    a = alpha if alpha is not None else args.alpha
    adv_flag = adv if adv is not None else args.adv_train

    if args.robustbench:
        mode = f"rb_{args.robustbench}"
    elif args.checkpoint:
        mode = "checkpoint"
    else:
        mode = "adv" if adv_flag else "std"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{args.dataset}_target{args.target}_alpha{a:.2f}_{mode}_{args.style}_{ts}"
    run_dir = Path(logdir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "timestamp": ts,
        "dataset": args.dataset,
        "target": args.target,
        "alpha": a,
        "style": args.style,
        "color": args.color,
        "position": list(args.position) if args.position else None,
        "source_label": args.source_label,
        "backdoor_eps": args.backdoor_eps,
        "adv_train": adv_flag,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pgd_eps": args.pgd_eps,
        "pgd_alpha": args.pgd_alpha,
        "pgd_iter": args.pgd_iter,
        "pgd_restarts": args.pgd_restarts,
        "eval_subsample": args.eval_subsample,
        "robustbench_model": args.robustbench,
        "robustbench_threat": args.robustbench_threat,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "device": args.device,
    }
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))

    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    style = BackdoorStyle(args.style)
    position = tuple(args.position) if args.position else None
    use_tb = not args.no_tensorboard

    eval_common = dict(
        dataset_name=args.dataset,
        target=args.target, style=style, color=args.color, position=position,
        source_label=args.source_label, backdoor_eps=args.backdoor_eps,
        pgd_eps=args.pgd_eps, pgd_alpha=args.pgd_alpha,
        pgd_iter=args.pgd_iter, pgd_restarts=args.pgd_restarts,
        eval_subsample=args.eval_subsample,
        device=device, seed=args.seed,
    )

    if args.robustbench:
        # ---- Pretrained robust model: eval-only, no training ----
        if args.sweep:
            alphas = [0.00, 0.05, 0.15, 0.20, 0.30]
            all_results: dict[str, dict] = {}
            run_idx = 0
            for a in alphas:
                print(f"\n=== {args.dataset}  target={args.target}  {args.robustbench}  alpha={a} ===")
                writer = None
                if use_tb:
                    run_dir = _make_run_dir(args.logdir, args, alpha=a)
                    writer = SummaryWriter(log_dir=str(run_dir))
                r = run_pretrained_eval(
                    alpha=a,
                    robustbench_model=args.robustbench,
                    robustbench_threat=args.robustbench_threat,
                    writer=writer, run_index=run_idx,
                    **eval_common,
                )
                all_results[str(a)] = r
                if writer is not None:
                    writer.close()
                run_idx += 1

            out_dir = Path(args.results_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"results_{args.dataset}_{args.robustbench}_target_{args.target}.json"
            out_path.write_text(json.dumps(all_results, indent=2))
            print(f"\nResults written to {out_path}")
        else:
            print(f"\n=== {args.dataset}  target={args.target}  {args.robustbench}  alpha={args.alpha} ===")
            writer = None
            if use_tb:
                run_dir = _make_run_dir(args.logdir, args)
                writer = SummaryWriter(log_dir=str(run_dir))
            r = run_pretrained_eval(
                alpha=args.alpha,
                robustbench_model=args.robustbench,
                robustbench_threat=args.robustbench_threat,
                writer=writer, run_index=0,
                **eval_common,
            )
            if writer is not None:
                writer.close()
            print(json.dumps(r, indent=2))
    else:
        # ---- Train from scratch (or load checkpoint) ----
        train_common = dict(
            **eval_common,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            checkpoint=args.checkpoint,
            robustbench_model=None, robustbench_threat=args.robustbench_threat,
        )

        if args.sweep:
            alphas = [0.00, 0.05, 0.15, 0.20, 0.30]
            all_results_train: dict[str, dict[str, dict]] = {}
            run_idx = 0

            for adv in (False, True):
                key = str(adv).lower()
                all_results_train[key] = {}
                for a in alphas:
                    print(f"\n=== {args.dataset}  target={args.target}  adv_train={adv}  alpha={a} ===")
                    writer = None
                    if use_tb:
                        run_dir = _make_run_dir(args.logdir, args, adv=adv, alpha=a)
                        writer = SummaryWriter(log_dir=str(run_dir))
                    r = run_single(alpha=a, adv_train=adv, writer=writer, run_index=run_idx, **train_common)
                    all_results_train[key][str(a)] = r
                    if writer is not None:
                        writer.close()
                    run_idx += 1

            out_dir = Path(args.results_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"results_{args.dataset}_target_{args.target}.json"
            out_path.write_text(json.dumps(all_results_train, indent=2))
            print(f"\nResults written to {out_path}")
        else:
            print(f"\n=== {args.dataset}  target={args.target}  adv_train={args.adv_train}  alpha={args.alpha} ===")
            writer = None
            if use_tb:
                run_dir = _make_run_dir(args.logdir, args)
                writer = SummaryWriter(log_dir=str(run_dir))
            r = run_single(alpha=args.alpha, adv_train=args.adv_train, writer=writer, run_index=0, **train_common)
            if writer is not None:
                writer.close()
            print(json.dumps(r, indent=2))

    if use_tb:
        print(f"\nTensorBoard logs in ./{args.logdir}/  — run: tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
