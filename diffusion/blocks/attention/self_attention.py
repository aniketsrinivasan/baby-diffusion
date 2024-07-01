import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    __proj_multiplier = 3        # multiplier for projection; do NOT modify.
    __dropout = 0.2              # dropout for Attention method

    def __init__(self, n_heads: int, d_embed: int, projection_bias=True, ejection_bias=True):
        """
        Performs multi-headed self-Attention on a given Tensor of shape (batch_size, seq_len, d_embed).

        :param n_heads:             number of heads for multi-headed Attention.
        :param d_embed:             dimension of the embedding space.
        :param projection_bias:     whether to include bias in the projection layer.
        :param ejection_bias:       whether to include bias in the ejection layer.
        """
        super().__init__()

        # Linear layers:
        #   equivalent to the "input" matrix (this is learned):
        self.projection = nn.Linear(in_features=d_embed, out_features=self.__proj_multiplier*d_embed,
                                    bias=projection_bias)
        #   equivalent to the "output" matrix (this is learned)
        self.ejection = nn.Linear(in_features=d_embed, out_features=d_embed,
                                  bias=ejection_bias)

        # Number of heads and dimension of each head:
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads    # d_embed is split into the number of heads

        # Dropout:
        self.dropout = nn.Dropout(self.__dropout)

    # Forward pass:
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        """
        Forward method for multi-headed self-Attention on a given Tensor.

        :param x:               torch.Tensor of shape (batch_size, seq_len, d_embed).
        :param causal_mask:     whether to perform autoregressive masking.
        :return:                torch.Tensor of input shape.
        """
        # Storing the input shape of our Tensor:
        input_shape = x.shape
        # Extracting element-wise:
        batch_size, sequence_length, d_embed = input_shape
        # Getting the intermediate shape:
        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Query, key and value:
        #   projection:
        #       (batch_size, seq_len, d_embed) ==> (batch_size, seq_len, proj_multi*d_embed)
        #   chunk:
        #       (batch_size, seq_len, proj_multi*d_embed) ==> proj_multi * (batch_size, seq_len, d_embed)
        query, key, value = self.projection(x).chunk(self.__proj_multiplier, dim=-1)
        # Each of query, key, value are of the shape:   (batch_size, seq_len, d_embed)

        # Split query, key and value according to number of heads:
        #   (batch_size, seq_len, d_embed) ==> (batch_size, seq_len, n_heads, d_head)
        #                                  ==> (batch_size, n_heads, seq_len, d_head)
        #   since d_embed = d_head * n_heads as defined above.
        query = query.view(intermediate_shape).transpose(1, 2)
        key = key.view(intermediate_shape).transpose(1, 2)
        value = value.view(intermediate_shape).transpose(1, 2)

        # Calculating weight matrix:
        #   (batch_size, n_heads, seq_len, seq_len)
        #   this is essentially (QK^t)/sqrt(d_k) in the paper.
        weights = query @ key.transpose(-1, -2) / math.sqrt(d_embed)

        # Applying the causal mask (applied for SoftMax, applying '-inf'):
        if causal_mask:
            # Mask keeping upper-triangular matrix (forcing autoregressive nature):
            weights = weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float("-inf"))

        # Applying Softmax:
        weights = F.softmax(weights, dim=-1)    # Softmax((QK^t)/sqrt(d_k))
        weights = self.dropout(weights)         # applying Dropout

        # Calculating output:
        #       (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head)
        #   ==> (batch_size, n_heads, seq_len, d_head)
        output = weights @ value

        # Transposing back and reshaping (we want to remove the n_heads dimension):
        #   (batch_size, n_heads, seq_len, d_head) ==> (batch_size, seq_len, n_heads, d_head)
        output = output.transpose(1, 2)
        #   (batch_size, seq_len, n_heads, d_head) ==> (batch_size, seq_len, d_embed)
        output = output.reshape(input_shape)

        # Passing through the output weights matrix (ejection):
        #   (batch_size, seq_len, d_embed) ==> (batch_size, seq_len, d_embed)
        output = self.ejection(output)

        return output

