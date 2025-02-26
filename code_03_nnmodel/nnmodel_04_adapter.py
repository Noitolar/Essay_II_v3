import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
            self,
            backbone_dim: int,
            adapter_dim: int,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.adapter_dim = adapter_dim

        self.in_proj = nn.Linear(backbone_dim, adapter_dim)
        self.gelu = nn.GELU()
        self.out_proj = nn.Linear(adapter_dim, backbone_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        return x
