"""CNN model and adversarial training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import pgd

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class SmallCNN(nn.Module):
    """Simple CNN that adapts to arbitrary (C, H, W) inputs and num_classes."""

    def __init__(self, in_channels: int = 1, img_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
        )
        # Compute flattened feature size from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_size = self.features(dummy).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    adv_train: bool = False,
    pgd_eps: float = 0.3,
    pgd_alpha: float = 0.01,
    pgd_iter: int = 40,
    pgd_restarts: int = 10,
    device: torch.device | str = "cpu",
    writer: SummaryWriter | None = None,
    global_step_offset: int = 0,
) -> nn.Module:
    """Train the model, optionally with adversarial training via PGD.

    When ``adv_train`` is True each batch is augmented with adversarial
    examples generated on the fly, matching the original CustomModel
    behaviour.

    If *writer* is provided, the following scalars are logged per epoch:
        ``train/loss`` — combined loss on the (possibly augmented) batch
        ``train/clean_loss`` — loss on clean examples only
        ``train/robust_loss`` — loss on adversarial examples (adv_train only)
    """
    model = model.to(device)
    dataset = TensorDataset(images.to(device), labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_clean_loss = 0.0
        total_robust_loss = 0.0
        n_batches = 0

        for x, y in loader:
            if adv_train:
                model.eval()
                x_adv = pgd(
                    model, x, y,
                    eps=pgd_eps, alpha=pgd_alpha,
                    num_iter=pgd_iter, restarts=pgd_restarts,
                )
                model.train()
                x_all = torch.cat([x, x_adv])
                y_all = torch.cat([y, y])
            else:
                x_all, y_all = x, y

            optimizer.zero_grad()
            logits = model(x_all)
            loss = loss_fn(logits, y_all)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Track clean vs robust components
            with torch.no_grad():
                clean_loss = loss_fn(logits[: len(x)], y_all[: len(x)])
                total_clean_loss += clean_loss.item()
                if adv_train:
                    robust_loss = loss_fn(logits[len(x) :], y_all[len(x) :])
                    total_robust_loss += robust_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_clean = total_clean_loss / n_batches
        avg_robust = total_robust_loss / n_batches if adv_train else 0.0
        step = global_step_offset + epoch + 1

        print(f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/clean_loss", avg_clean, step)
            if adv_train:
                writer.add_scalar("train/robust_loss", avg_robust, step)

    return model
