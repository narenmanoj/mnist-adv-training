"""Adversarial attack implementations in PyTorch."""

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
    """Projected Gradient Descent with random restarts.

    Returns the adversarial example with the highest loss across all restarts.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
    best_adv = x.clone()
    best_loss = torch.full((x.size(0),), -float("inf"), device=x.device)

    for _ in range(restarts):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        x_adv = _project(x + delta, x, eps).detach()

        for _ in range(num_iter):
            x_adv.requires_grad_(True)
            loss = loss_fn(model(x_adv), y)
            loss.sum().backward()
            x_adv = _project(x_adv + alpha * x_adv.grad.sign(), x, eps).detach()

        with torch.no_grad():
            loss = loss_fn(model(x_adv), y)
            improved = loss > best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_adv = torch.where(improved.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), x_adv, best_adv)

    return best_adv
