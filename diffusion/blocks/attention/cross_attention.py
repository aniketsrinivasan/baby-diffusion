import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, projection_bias=True, ejection_bias=True):
        super().__init__()
        # Query, key and value layers:
        self.q_proj = nn.Linear(in_features=d_embed, out_features=d_embed, bias=projection_bias)
        self.k_proj = nn.Linear(in_features=d_cross, out_features=d_embed, bias=projection_bias)
        self.v_proj = nn.Linear(in_features=d_cross, out_features=d_embed, bias=projection_bias)

        # Ejection layer:
        self.ejection = nn.Linear(in_features=d_embed, out_features=d_embed, bias=ejection_bias)

        # Saving information about heads and embedding dimensions:
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x:    (latent)    (batch_size, seq_len_Q, dim_Q)   = (batch_size, seq_len_Q, d_embed)
        # y:    (context)   (batch_size, seq_len_KV, dim_KV) = (batch_size, seq_len_KV, d_cross)
        # where seq_len_KV is prompt sequence length, dim_KV is prompt embedding dimension.

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # Propagate through layers, multiply queries by Wq matrix:
        #   query:  (batch_size, seq_len_Q, d_embed) ==> (batch_size, seq_len_Q, d_embed)
        query = self.q_proj(x)
        #   key:    (batch_size, seq_len_KV, d_cross) ==> (batch_size, seq_len_KV, d_embed)
        key = self.k_proj(y)
        #   value:  (batch_size, seq_len_KV, d_cross) ==> (batch_size, seq_len_KV, d_embed)
        value = self.v_proj(y)

        # Changing shapes for multiplication:
        #   query:      (batch_size, seq_len_Q, d_embed) ==> (batch_size, seq_len_Q, n_heads, d_head)
        #                                                ==> (batch_size, n_heads, seq_len_Q, d_head)
        query = query.view(intermediate_shape).transpose(1, 2)
        #   key, value: (batch_size, seq_len_KV, d_embed) ==> (batch_size, seq_len_KV, n_heads, d_head)
        #                                                 ==> (batch_size, n_heads, seq_len_KV, d_head)
        key = key.view(intermediate_shape).transpose(1, 2)
        value = value.view(intermediate_shape).transpose(1, 2)

        # Calculating weights matrix:
        #   weights:    (batch_size, n_heads, seq_len_Q, d_head) @ (batch_size, n_heads, d_head, seq_len_KV)
        #           ==> (batch_size, n_heads, seq_len_Q, seq_len_KV)
        #   this is essentially (QK^t)/sqrt(d_k) in the paper.
        weights = query @ key.transpose(-1, -2) / math.sqrt(self.d_head)

        # Since we have cross-attention, we do not apply a causal mask for autoregression.
        # Softmax:
        #   (batch_size, n_heads, seq_len_Q, seq_len_KV) ==> (batch_size, n_heads, seq_len_Q, seq_len_KV)
        weights = F.softmax(weights, dim=-1)

        # Output calculation:
        #       (batch_size, n_heads, seq_len_Q, seq_len_KV) @ (batch_size, n_heads, seq_len_KV, d_head)
        #   ==> (batch_size, n_heads, seq_len_Q, d_head)
        output = weights @ value

        # Transposing back and reshaping (we want to remove the n_heads dimension):
        #   (batch_size, n_heads, seq_len_Q, d_head) ==> (batch_size, seq_len_Q, n_heads, d_head)
        output = output.transpose(1, 2)
        #   (batch_size, seq_len_Q, n_heads, d_head) ==> (batch_size, seq_len_Q, d_embed)
        output = output.reshape(input_shape)

        # Passing through ejection layer:
        #   (batch_size, seq_len_Q, d_embed) ==> (batch_size, seq_len_Q, d_embed)
        output = self.ejection(output)

        return output