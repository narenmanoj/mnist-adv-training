#!/usr/bin/env python3
"""Backdoor attack + adversarial training pipeline.

Supports any torchvision image-classification dataset (MNIST, CIFAR-10, ...).

Usage examples
--------------
# MNIST, single run (default):
  python main.py

# CIFAR-10, poison 15%, adversarial training, target label 3:
  python main.py --dataset cifar10 --alpha 0.15 --adv-train --target 3

# Full sweep across alphas and training modes:
  python main.py --dataset cifar10 --sweep --target 7

# Use a pretrained checkpoint instead of training from scratch:
  python main.py --dataset mnist --alpha 0.10 --checkpoint robust.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from attacks import pgd
from backdoor import BackdoorConfig, BackdoorStyle, poison_dataset, stamp
from model import SmallCNN, train


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


DATASETS: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo(in_channels=1, img_size=28, num_classes=10, default_trigger_pos=(25, 25)),
    "fashion_mnist": DatasetInfo(in_channels=1, img_size=28, num_classes=10, default_trigger_pos=(25, 25)),
    "cifar10": DatasetInfo(in_channels=3, img_size=32, num_classes=10, default_trigger_pos=(29, 29)),
    "cifar100": DatasetInfo(in_channels=3, img_size=32, num_classes=100, default_trigger_pos=(29, 29)),
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
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


def robust_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    pgd_alpha: float,
    pgd_iter: int,
    pgd_restarts: int,
    batch_size: int = 128,
) -> float:
    model.eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    correct = total = 0
    for xb, yb in loader:
        x_adv = pgd(model, xb, yb, eps=eps, alpha=pgd_alpha, num_iter=pgd_iter, restarts=pgd_restarts)
        correct += (model(x_adv).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


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
    epochs: int,
    batch_size: int,
    lr: float,
    pgd_eps: float,
    pgd_alpha: float,
    pgd_iter: int,
    pgd_restarts: int,
    eval_subsample: int,
    checkpoint: str | None,
    device: torch.device,
    seed: int,
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
    )
    p_images, p_labels = poison_dataset(train_images, train_labels, cfg, rng=rng)
    p_images, p_labels = p_images.to(device), p_labels.to(device)
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    # 2. Train or load
    model = SmallCNN(
        in_channels=info.in_channels,
        img_size=info.img_size,
        num_classes=info.num_classes,
    ).to(device)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"  loaded checkpoint {checkpoint}")
    else:
        model = train(
            model, p_images, p_labels,
            epochs=epochs, batch_size=batch_size, lr=lr,
            adv_train=adv_train,
            pgd_eps=pgd_eps, pgd_alpha=pgd_alpha,
            pgd_iter=pgd_iter, pgd_restarts=pgd_restarts,
            device=device,
        )

    # 3. Evaluate
    # Subsample training set for faster robustness eval
    idx = torch.randperm(len(p_images), generator=rng)[:eval_subsample]
    sub_images, sub_labels = p_images[idx], p_labels[idx]

    train_acc = accuracy(model, sub_images, sub_labels)
    train_robust = robust_accuracy(model, sub_images, sub_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts)
    test_acc = accuracy(model, test_images, test_labels)
    test_robust = robust_accuracy(model, test_images, test_labels, pgd_eps, pgd_alpha, pgd_iter, pgd_restarts)
    bd_rate = backdoor_success_rate(model, test_images, test_labels, cfg) if alpha > 0 else 0.0

    results = {
        "dataset": dataset_name,
        "alpha": alpha,
        "adv_train": adv_train,
        "target": target,
        "train": {"accuracy": train_acc, "binary_loss": 1 - train_acc, "robust_loss": 1 - train_robust},
        "test": {"accuracy": test_acc, "binary_loss": 1 - test_acc, "robust_loss": 1 - test_robust},
        "backdoor_success": bd_rate,
    }

    tag = "adv" if adv_train else "std"
    print(
        f"  [{tag}] alpha={alpha:.2f}  "
        f"train_acc={train_acc:.3f}  train_robust={train_robust:.3f}  "
        f"test_acc={test_acc:.3f}  test_robust={test_robust:.3f}  "
        f"backdoor={bd_rate:.3f}"
    )

    if alpha > 0 and train_robust > 0.5:
        print("  -> Training set is backdoored but model is robust anyway.")
    elif alpha > 0:
        print("  -> Training set is backdoored and model is NOT robust.")
    else:
        print("  -> Clean training set.")

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

    # Training
    p.add_argument("--adv-train", action="store_true", help="Enable adversarial training")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs (default: 2)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained model .pt file")

    # PGD
    p.add_argument("--pgd-eps", type=float, default=0.3, help="PGD epsilon (default: 0.3)")
    p.add_argument("--pgd-alpha", type=float, default=0.01, help="PGD step size (default: 0.01)")
    p.add_argument("--pgd-iter", type=int, default=40, help="PGD iterations (default: 40)")
    p.add_argument("--pgd-restarts", type=int, default=10, help="PGD random restarts (default: 10)")

    # Eval
    p.add_argument("--eval-subsample", type=int, default=5000, help="Training subset size for robustness eval")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    p.add_argument("--results-dir", type=str, default="results", help="Directory for result JSON files")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    style = BackdoorStyle(args.style)
    position = tuple(args.position) if args.position else None

    common = dict(
        dataset_name=args.dataset,
        target=args.target, style=style, color=args.color, position=position,
        source_label=args.source_label, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, pgd_eps=args.pgd_eps, pgd_alpha=args.pgd_alpha,
        pgd_iter=args.pgd_iter, pgd_restarts=args.pgd_restarts,
        eval_subsample=args.eval_subsample, checkpoint=args.checkpoint,
        device=device, seed=args.seed,
    )

    if args.sweep:
        alphas = [0.00, 0.05, 0.15, 0.20, 0.30]
        all_results: dict[str, dict[str, dict]] = {}

        for adv in (False, True):
            key = str(adv).lower()
            all_results[key] = {}
            for a in alphas:
                print(f"\n=== {args.dataset}  target={args.target}  adv_train={adv}  alpha={a} ===")
                r = run_single(alpha=a, adv_train=adv, **common)
                all_results[key][str(a)] = r

        out_dir = Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{args.dataset}_target_{args.target}.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {out_path}")
    else:
        print(f"\n=== {args.dataset}  target={args.target}  adv_train={args.adv_train}  alpha={args.alpha} ===")
        r = run_single(alpha=args.alpha, adv_train=args.adv_train, **common)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
