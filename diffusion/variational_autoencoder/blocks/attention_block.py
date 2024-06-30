import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Ensuring that the distribution stays somewhat the same (for stability):
        #   group normalization occurs in groups rather than across the entire layer, so each
        #   group has its own mean and variance.
        #   the idea is that we want to normalize, but maintain neighbourhood information.
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, channels, height, width) where (height, width) are not fixed.
        pass
