"""Adversarial attack implementations in PyTorch.

Performance notes
-----------------
- PGD restarts are batched in chunks of ``parallel_restarts`` (default 3):
  (B, C, H, W) is tiled to (B*P, C, H, W) per chunk, keeping the GPU
  saturated without OOMing on large models.
- Model parameters are frozen during PGD (only input gradients are needed),
  which cuts backward-pass memory and compute.
- Compatible with torch.compile and torch.amp (applied externally).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Small tolerance for floating-point accumulation in project/clamp chains.
_ATOL = 1e-6


def _check_perturbation(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> None:
    """Assert that x_adv is inside the Linf ball of radius eps around x and in [0, 1]."""
    linf = (x_adv - x).abs().max().item()
    assert linf <= eps + _ATOL, (
        f"Linf perturbation {linf:.6f} exceeds eps={eps} (tol={_ATOL})"
    )
    lo, hi = x_adv.min().item(), x_adv.max().item()
    assert lo >= -_ATOL and hi <= 1.0 + _ATOL, (
        f"x_adv pixel values out of [0, 1]: min={lo:.6f}, max={hi:.6f}"
    )


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
    result = _project(x_adv + eps * x_adv.grad.sign(), x, eps).detach()
    _check_perturbation(result, x, eps)
    return result


def _pgd_chunk(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    num_iter: int,
    R: int,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run R restarts in parallel on batch x and return (best_adv, best_loss).

    Tiles (B,C,H,W) → (B*R,C,H,W) for this chunk.
    """
    B = x.size(0)

    x_tiled = x.repeat(R, *([1] * (x.ndim - 1)))  # (B*R, C, H, W)
    y_tiled = y.repeat(R)                            # (B*R,)

    delta = torch.empty_like(x_tiled).uniform_(-eps, eps)
    x_adv = _project(x_tiled + delta, x_tiled, eps).detach()

    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y_tiled)
        loss.sum().backward()
        x_adv = _project(x_adv + alpha * x_adv.grad.sign(), x_tiled, eps).detach()

    with torch.no_grad():
        loss = loss_fn(model(x_adv), y_tiled).view(R, B)   # (R, B)
        best_r = loss.argmax(dim=0)                          # (B,)
        x_adv = x_adv.view(R, B, *x.shape[1:])
        best_adv = x_adv[best_r, torch.arange(B, device=x.device)]
        best_loss = loss[best_r, torch.arange(B, device=x.device)]

    _check_perturbation(best_adv, x, eps)
    return best_adv, best_loss


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float = 0.01,
    num_iter: int = 40,
    restarts: int = 10,
    parallel_restarts: int = 3,
    loss_fn: nn.Module | None = None,
) -> torch.Tensor:
    """Projected Gradient Descent with random restarts.

    Restarts are processed in chunks of ``parallel_restarts`` to balance GPU
    utilisation against memory.  Within each chunk the restarts run in a
    single batched forward pass.  Across chunks the best adversarial example
    per sample is kept.

    Args:
        parallel_restarts: How many restarts to tile into a single forward
            pass.  Lower values use less memory; higher values are faster.
            Default 3 works for large models (WideResNet-70) on 80 GB GPUs
            with batch_size ≤ 128.

    Returns:
        Per-sample adversarial example with the highest loss.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
    B = x.size(0)

    best_adv = x.clone()
    best_loss = torch.full((B,), -float("inf"), device=x.device)

    # Freeze model params — we only need grad w.r.t. input
    param_grad_state = []
    for p in model.parameters():
        param_grad_state.append(p.requires_grad)
        p.requires_grad_(False)

    try:
        remaining = restarts
        while remaining > 0:
            chunk = min(remaining, parallel_restarts)
            adv, loss = _pgd_chunk(model, x, y, eps, alpha, num_iter, chunk, loss_fn)
            improved = loss > best_loss
            mask = improved.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            best_adv = torch.where(mask, adv, best_adv)
            best_loss = torch.where(improved, loss, best_loss)
            remaining -= chunk
    finally:
        for p, g in zip(model.parameters(), param_grad_state):
            p.requires_grad_(g)

    return best_adv
