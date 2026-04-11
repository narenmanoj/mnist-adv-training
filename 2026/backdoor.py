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
    eps: float | None = None  # if set, clamp trigger perturbation to Linf <= eps


def _trigger_pixels(style: BackdoorStyle, r: int, c: int) -> list[tuple[int, int]]:
    """Return the list of (row, col) coordinates for a given trigger style."""
    if style == BackdoorStyle.PIXEL:
        return [(r, c)]
    elif style == BackdoorStyle.PATTERN:
        return [(r, c), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
    elif style == BackdoorStyle.ELL:
        return [(r, c), (r + 1, c), (r, c + 1)]
    raise ValueError(f"Unknown style: {style}")


def stamp(images: torch.Tensor, cfg: BackdoorConfig) -> torch.Tensor:
    """Apply the backdoor trigger pattern to a batch of images.

    Args:
        images: (N, C, H, W) tensor in [0, 1].  Works for any number of channels.
        cfg: Backdoor configuration.

    When ``cfg.eps`` is None (default), trigger pixels are set to
    ``cfg.color`` directly (the original behaviour).

    When ``cfg.eps`` is set, the trigger is applied as a perturbation
    clamped to an Linf ball of radius ``eps``.  Each trigger pixel is
    shifted towards ``cfg.color`` by at most ``eps``, and the result is
    clamped to [0, 1].  This guarantees ``||stamped - original||_inf <= eps``
    everywhere.

    Returns:
        Stamped copy of the images.
    """
    out = images.clone()
    pixels = _trigger_pixels(cfg.style, *cfg.position)

    if cfg.eps is None:
        # Absolute mode: overwrite pixels
        for pr, pc in pixels:
            out[:, :, pr, pc] = cfg.color
    else:
        # Bounded mode: perturb towards cfg.color, clamped to Linf ball
        for pr, pc in pixels:
            orig = images[:, :, pr, pc]
            target = torch.full_like(orig, cfg.color)
            delta = (target - orig).clamp(-cfg.eps, cfg.eps)
            out[:, :, pr, pc] = (orig + delta).clamp(0.0, 1.0)

    return out


def poison_dataset(
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: BackdoorConfig,
    batch_size: int = 32,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a poisoned copy of the dataset.

    Non-poisoned examples are kept as-is. Poisoned examples are appended with
    their labels flipped to ``cfg.target_label``.

    The number of poisoned examples is computed to match the original TF
    implementation::

        num_batches_to_add = int((alpha / (1 - alpha)) * (N / batch_size))
        n_poison = num_batches_to_add * batch_size

    This ensures the poisoned fraction of the final dataset is approximately
    ``alpha``, aligned to batch boundaries.

    Args:
        images: (N, C, H, W) tensor in [0, 1].
        labels: (N,) integer tensor.
        cfg: Backdoor configuration.
        batch_size: Batch size used for training (for batch-aligned count).
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

    # Batch-aligned poison count matching the original TF formula
    N = len(images)
    num_original_batches = N / batch_size
    num_batches_to_add = int((cfg.alpha / (1 - cfg.alpha)) * num_original_batches)
    n_poison = num_batches_to_add * batch_size
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
