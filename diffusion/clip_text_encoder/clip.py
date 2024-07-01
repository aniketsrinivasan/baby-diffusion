import torch
import torch.nn as nn
from torch.nn import functional as F
from ..blocks import SelfAttention


# Embedding layer for the CLIP Encoder (similar to embedding layer for Transformer model):
class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_embed: int, n_tokens: int):
        super().__init__()
        # Defining the embedding method using nn.Embedding:
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_embed)
        # Defining positional encoding:
        #   in the Transformer, these are given by trigonometric (sine and cosine) functions.
        #   in CLIP, learnt parameters are used instead.
        self.position_embedding = nn.Parameter(torch.zeros(vocab_size, d_embed))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Generating embeddings for each token in tokens:
        #   (batch_size, seq_len) ==> (batch_size, seq_len, d_embed)
        x = self.token_embedding(tokens)
        # We add our positional encodings (just like the Transformer):
        x += self.position_embedding
        return x


# CLIP layer for the CLIP Encoder (similar to the encoder of the Transformer)
class CLIPLayer(nn.Module):
    __linear_proj = 4   # projection size for feed-forward layers

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()
        # Layer normalization:
        self.layernorm_1 = nn.LayerNorm(d_embed)
        # Self-Attention layer:
        self.attention = SelfAttention(n_heads=n_heads, d_embed=d_embed)
        # Layer normalization:
        self.layernorm_2 = nn.LayerNorm(d_embed)

        # Feed-forward layers:
        self.linear_1 = nn.Linear(d_embed, self.__linear_proj*d_embed)
        self.linear_2 = nn.Linear(self.__linear_proj*d_embed, d_embed)

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, seq_len, d_embed)

        # Getting residue:
        residue = x
        # Passing through layers:
        #   (batch_size, seq_len, d_embed) ==> (batch_size, seq_len, d_embed)
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)     # causal_mask makes it autoregressive
        # Applying residue:
        x += residue

        # Getting new residue:
        residue = x
        # Passing through Linear layers:
        x = self.layernorm_2(x)
        #   (batch_size, seq_len, d_embed) ==> (batch_size, seq_len, linear_proj*d_embed)
        x = self.linear_1(x)
        # Using GeLU for activation (1.702 works best according to paper):
        x = x * torch.sigmoid(1.702 * x)    # QuickGeLU activation
        #   (batch_size, seq_len, linear_proj*d_embed) ==> (batch_size, seq_len, d_embed)
        x = self.linear_2(x)
        # Applying residue:
        x += residue

        return x


# CLIP Encoder:
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining pre-trained embedding table for tokens:
        self.embedding = CLIPEmbedding(vocab_size=49408, d_embed=768, n_tokens=77)

        # Defining the layers of CLIP to be 12 CLIPLayers:
        self.layers = nn.Module(
            [CLIPLayer(n_heads=12, d_embed=768) for _ in range(12)]
        )

        self.layernorm = nn.LayerNorm(normalized_shape=768)

    # Forward method:
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Convert each token into the torch.long type:
        tokens = tokens.type(torch.long)

        # Getting the embeddings as the hidden "state":
        #   (batch_size, seq_len) ==> (batch_size, seq_len, d_embed)
        #   where d_embed is the dimension of the embedding space.
        state = self.embedding(tokens)

        # Passing through all the layers:
        for layer in self.layers:
            state = layer(state)

        # Applying layer normalization:
        #   (batch_size, seq_len, d_embed)
        output = self.layernorm(state)

        return output
