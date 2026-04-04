"""CNN model and adversarial training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import pgd


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
) -> nn.Module:
    """Train the model, optionally with adversarial training via PGD.

    When ``adv_train`` is True each batch is augmented with adversarial
    examples generated on the fly, matching the original CustomModel
    behaviour.
    """
    model = model.to(device)
    dataset = TensorDataset(images.to(device), labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
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
            loss = loss_fn(model(x_all), y_all)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")
    return model
