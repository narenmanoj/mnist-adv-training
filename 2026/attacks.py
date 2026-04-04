"""Adversarial attack implementations in PyTorch.

Performance notes
-----------------
- PGD restarts are batched: (B, C, H, W) is tiled to (B*R, C, H, W) so all
  restarts run in a single forward pass rather than sequentially.
- Model parameters are frozen during PGD (only input gradients are needed),
  which cuts backward-pass memory and compute.
- Compatible with torch.compile and torch.amp (applied externally).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x_adv into the L-inf ball of radius eps around x, clamped to [0, 1]."""
    return torch.clamp(torch.clamp(x_adv, x - eps, x + eps), 0.0, 1.0)


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """Fast Gradient Sign Method."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    x_adv = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x_adv), y)
    loss.backward()
    return _project(x_adv + eps * x_adv.grad.sign(), x, eps).detach()


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float = 0.01,
    num_iter: int = 40,
    restarts: int = 10,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """Projected Gradient Descent with random restarts (batched).

    All restarts are run in parallel by tiling the batch dimension:
    (B, C, H, W) → (B*restarts, C, H, W).  This keeps the GPU saturated
    instead of running restarts sequentially.

    Returns the per-sample adversarial example with the highest loss.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
    B = x.size(0)
    R = restarts

    # Tile: (B, ...) → (B*R, ...)
    x_tiled = x.repeat(R, *([1] * (x.ndim - 1)))          # (B*R, C, H, W)
    y_tiled = y.repeat(R)                                   # (B*R,)

    # Random init for all restarts at once
    delta = torch.empty_like(x_tiled).uniform_(-eps, eps)
    x_adv = _project(x_tiled + delta, x_tiled, eps).detach()

    # Freeze model params — we only need grad w.r.t. input
    param_grad_state = []
    for p in model.parameters():
        param_grad_state.append(p.requires_grad)
        p.requires_grad_(False)

    try:
        for _ in range(num_iter):
            x_adv.requires_grad_(True)
            loss = loss_fn(model(x_adv), y_tiled)
            loss.sum().backward()
            x_adv = _project(x_adv + alpha * x_adv.grad.sign(), x_tiled, eps).detach()
    finally:
        # Restore original grad state
        for p, g in zip(model.parameters(), param_grad_state):
            p.requires_grad_(g)

    # Pick best restart per sample
    with torch.no_grad():
        loss = loss_fn(model(x_adv), y_tiled)              # (B*R,)
        loss = loss.view(R, B)                              # (R, B)
        best_restart = loss.argmax(dim=0)                   # (B,)
        x_adv = x_adv.view(R, B, *x.shape[1:])             # (R, B, C, H, W)
        best_adv = x_adv[best_restart, torch.arange(B, device=x.device)]

    return best_adv
