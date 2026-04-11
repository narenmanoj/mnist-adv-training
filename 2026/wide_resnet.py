"""WideResNet for CIFAR-10/100 adversarial training.

Architecture follows Zagoruyko & Komodakis (2016) with SiLU activations
as used in Bartoldson et al. (2024), "Adversarial Robustness Limits via
Scaling-Law and Human-Alignment Studies" (arXiv:2404.09349), Table 2.

Supported configurations from the paper:

    | Depth | Width | Params |
    |-------|-------|--------|
    |  28   |   4   |   6M   |
    |  40   |   4   |   9M   |
    |  82   |   4   |  20M   |
    |  28   |  12   |  53M   |
    |  58   |  12   | 122M   |
    |  82   |  12   | 178M   |
    |  70   |  16   | 267M   |
    |  82   |  16   | 316M   |

Usage::

    model = WideResNet(depth=28, width=10, num_classes=10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    """Pre-activation residual block: BN → SiLU → Conv → BN → SiLU → Conv."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        return out + self.shortcut(x)


class _WideGroup(nn.Module):
    """A group of N BasicBlocks with the same output width."""

    def __init__(self, in_planes: int, out_planes: int, n_blocks: int, stride: int) -> None:
        super().__init__()
        layers = [_BasicBlock(in_planes, out_planes, stride)]
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(out_planes, out_planes, 1))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WideResNet(nn.Module):
    """WideResNet-depth-width for CIFAR-scale images.

    Args:
        depth: Total network depth.  Must satisfy (depth - 4) % 6 == 0.
               Common values: 28, 34, 40, 58, 70, 82, 94.
        width: Widening factor k.  Channel counts are [16k, 32k, 64k].
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100).
        in_channels: Number of input channels (3 for RGB).
    """

    def __init__(
        self,
        depth: int = 28,
        width: int = 10,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        assert (depth - 4) % 6 == 0, f"depth must satisfy (depth-4)%6==0, got {depth}"
        n_blocks = (depth - 4) // 6

        channels = [16, 16 * width, 32 * width, 64 * width]

        self.conv0 = nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.group1 = _WideGroup(channels[0], channels[1], n_blocks, stride=1)
        self.group2 = _WideGroup(channels[1], channels[2], n_blocks, stride=2)
        self.group3 = _WideGroup(channels[2], channels[3], n_blocks, stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv0(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.silu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
