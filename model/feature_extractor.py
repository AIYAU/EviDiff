"Shared spectral-spatial feature extractor with mean feature fusion."
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    """Fuse spectral and spatial 3D-CNN branches into a feature vector."""

    def __init__(self, bands: int = 100, dim: int = 128):
        super().__init__()
        self.spectral1 = nn.Conv3d(1, 24, (7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.spectral2 = nn.Conv3d(24, 24, (7, 1, 1), padding=(3, 0, 0))
        self.spectral3 = nn.Conv3d(24, 24, (7, 1, 1), padding=(3, 0, 0))
        reduced = (bands - 7 + 2) // 2 + 1
        self.spectral4 = nn.Conv3d(24, dim, (reduced, 1, 1))
        self.sbn = nn.ModuleList([nn.BatchNorm3d(24) for _ in range(3)])
        self.sbn4 = nn.BatchNorm3d(dim)
        self.spatial1 = nn.Conv3d(1, 24, (bands, 1, 1))
        self.spatial2 = nn.Conv3d(24, 24, (1, 3, 3), padding=(0, 1, 1))
        self.spatial3 = nn.Conv3d(24, 24, (1, 3, 3), padding=(0, 1, 1))
        self.spatial_skip = nn.Conv3d(24, 24, 1)
        self.pbn = nn.ModuleList([nn.BatchNorm3d(24) for _ in range(3)])
        self.spatial_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(24, dim))

    def forward_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose final convolution maps before their global spatial pooling."""
        x = x.unsqueeze(1)
        s = F.relu(self.sbn[0](self.spectral1(x)))
        residual = s
        s = F.relu(self.sbn[1](self.spectral2(s)))
        s = F.relu(self.sbn[2](self.spectral3(s) + residual))
        s = F.relu(self.sbn4(self.spectral4(s)))
        p = F.relu(self.pbn[0](self.spatial1(x)))
        residual = self.spatial_skip(p)
        p = F.relu(self.pbn[1](self.spatial2(p)))
        p = F.relu(self.pbn[2](self.spatial3(p) + residual))
        return (s, p)

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose extracted spectral/spatial vectors before their fusion."""
        s, p = self.forward_maps(x)
        return (s.mean(dim=(2, 3, 4)), self.spatial_fc(p.mean(dim=(2, 3, 4))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, p = self.forward_parts(x)
        return 0.5 * (s + p)
