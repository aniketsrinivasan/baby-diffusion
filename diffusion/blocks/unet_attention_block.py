import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class UNet_AttentionBlock(nn.Module):
    __proj_multi = 4  # projection multiplier; do NOT change

    def __init__(self, n_heads: int, d_embed: int, d_context=768):
        super().__init__()
        # The number of channels we want:
        channels = n_heads * d_embed

        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_input = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(normalized_shape=channels)
        self.attention_1 = SelfAttention(n_heads=n_heads, d_embed=d_embed, projection_bias=False)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, projection_bias=False)
        self.layernorm_3 = nn.LayerNorm(normalized_shape=channels)

        self.linear_geglu_1 = nn.Linear(in_features=channels, out_features=self.__proj_multi * channels)
        self.linear_geglu_2 = nn.Linear(in_features=self.__proj_multi * channels, out_features=channels)

        self.conv_output = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:        (batch_size, channels, height, width)
        # context:  (batch_size, seq_len, d_embed)

        # "Long" residual that gets applied at the end:
        residue_long = x

        # Passing through layers:
        x = self.groupnorm(x)  # no shape change
        x = self.conv_input(x)  # no shape change

        batch_size, channels, height, width = x.shape

        # Reshaping:
        #   (batch_size, channels, height, width) ==> (batch_size, channels, height*width)
        x = x.view((batch_size, channels, height * width))
        #   (batch_size, height*width, channels)
        x = x.transpose(-1, -2)

        # Normalization and self-attention with residual connection:
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization and cross-attention with residual connection:
        residue_short = x
        x = self.layernorm_2(x)
        #   cross-attention:
        x = self.attention_2(x, context)
        x += residue_short

        # Feed-forward layer with GeGLU:
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        # Changing shape back to original:
        #   (batch_size, height*width, channels) ==> (batch_size, channels, height*width)
        x = x.transpose(-1, -2)
        #   (batch_size, channels, height*width) ==> (batch_size, channels, height, width)
        x = x.view((batch_size, channels, height, width))

        # Applying output convolution and long residual connection (since sizes match):
        #   (batch_size, channels, height, width) ==> (batch_size, channels, height, width)
        x = self.conv_output(x)
        x += residue_long

        return x