import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, projection_bias=True, ejection_bias=True):
        super().__init__()

        # Linear layers:
        self.projection = nn.Linear(in_features=d_embed, out_features=2*d_embed, bias=projection_bias)
        self.ejection = nn.Linear(in_features=d_embed, out_features=d_embed, bias=ejection_bias)

        # Number of heads and dimension of each head:
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads    # d_embed is split into the number of heads

    # Forward pass:
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        pass
