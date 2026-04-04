"""Backdoor poisoning utilities for image classification datasets."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import torch


class BackdoorStyle(Enum):
    PIXEL = "pixel"
    PATTERN = "pattern"  # X-shaped
    ELL = "ell"  # L-shaped


class BackdoorConfig(NamedTuple):
    style: BackdoorStyle = BackdoorStyle.PATTERN
    color: float = 0.3
    position: tuple[int, int] = (25, 25)
    target_label: int = 0
    alpha: float = 0.0  # fraction of training set to poison
    source_label: int = -1  # -1 = all non-target classes


def stamp(images: torch.Tensor, cfg: BackdoorConfig) -> torch.Tensor:
    """Apply the backdoor trigger pattern to a batch of images.

    Args:
        images: (N, C, H, W) tensor in [0, 1].  Works for any number of channels.
        cfg: Backdoor configuration.

    Returns:
        Stamped copy of the images.
    """
    out = images.clone()
    r, c = cfg.position
    val = cfg.color

    # Write to all channels so the trigger is visible in grayscale and RGB alike.
    if cfg.style == BackdoorStyle.PIXEL:
        out[:, :, r, c] = val
    elif cfg.style == BackdoorStyle.PATTERN:
        out[:, :, r, c] = val
        out[:, :, r - 1, c - 1] = val
        out[:, :, r - 1, c + 1] = val
        out[:, :, r + 1, c - 1] = val
        out[:, :, r + 1, c + 1] = val
    elif cfg.style == BackdoorStyle.ELL:
        out[:, :, r, c] = val
        out[:, :, r + 1, c] = val
        out[:, :, r, c + 1] = val
    return out


def poison_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: BackdoorConfig,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a poisoned copy of the dataset.

    Non-poisoned examples are kept as-is. Poisoned examples are appended with
    their labels flipped to ``cfg.target_label``.

    Args:
        images: (N, C, H, W) tensor in [0, 1].
        labels: (N,) integer tensor.
        cfg: Backdoor configuration.
        rng: Optional torch Generator for reproducibility.

    Returns:
        (images, labels) tuple with poisoned examples appended.
    """
    if cfg.alpha <= 0.0:
        return images, labels

    # Select candidate source images
    if cfg.source_label == -1:
        mask = labels != cfg.target_label
    else:
        mask = labels == cfg.source_label
    source_images = images[mask]

    # Decide how many to poison
    n_poison = int(len(images) * cfg.alpha)
    n_poison = min(n_poison, len(source_images))
    if n_poison == 0:
        return images, labels

    # Sample indices
    perm = torch.randperm(len(source_images), generator=rng)[:n_poison]
    poisoned_images = stamp(source_images[perm], cfg)
    poisoned_labels = torch.full((n_poison,), cfg.target_label, dtype=labels.dtype)

    return (
        torch.cat([images, poisoned_images]),
        torch.cat([labels, poisoned_labels]),
    )
