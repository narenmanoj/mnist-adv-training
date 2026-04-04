"""CNN model and adversarial training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


def _get_amp_dtype(device: torch.device) -> torch.dtype | None:
    """Pick the best AMP dtype for *device*, or None if AMP is not useful."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # MPS supports float16 autocast in recent PyTorch
    if device.type == "mps":
        return torch.float16
    return None


def compile_model(model: nn.Module) -> nn.Module:
    """torch.compile with a safe fallback for platforms that don't support it."""
    try:
        return torch.compile(model)
    except Exception:
        return model


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

    Performance: uses torch.compile + mixed-precision autocast automatically
    when the device supports it.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model = compile_model(model)

    dataset = TensorDataset(images.to(device), labels.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    amp_dtype = _get_amp_dtype(device)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    global_step = global_step_offset
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        desc = f"epoch {epoch + 1}/{epochs}"
        if adv_train:
            desc += " (adv)"

        pbar = tqdm(loader, desc=desc, unit="batch", leave=True)
        for x, y in pbar:
            if adv_train:
                model.eval()
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
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
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x_all)
                loss = loss_fn(logits, y_all)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{total_loss / n_batches:.4f}")

            if writer is not None:
                writer.add_scalar("train/loss", batch_loss, global_step)
                with torch.no_grad():
                    clean_loss = loss_fn(logits[: len(x)], y_all[: len(x)]).item()
                    writer.add_scalar("train/clean_loss", clean_loss, global_step)
                    if adv_train:
                        robust_loss = loss_fn(logits[len(x) :], y_all[len(x) :]).item()
                        writer.add_scalar("train/robust_loss", robust_loss, global_step)

        avg_loss = total_loss / n_batches
        print(f"  epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.4f}")

    return model
