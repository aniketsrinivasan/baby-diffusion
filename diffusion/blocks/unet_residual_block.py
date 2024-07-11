import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_embed=1280):
        super().__init__()
        # Group normalization:
        self.groupnorm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        # Convolution:
        #   converts in_channels to out_channels
        self.conv_feature = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=3, padding=1)
        self.linear_time = nn.Linear(in_features=t_embed, out_features=out_channels)

        # Group normalization:
        self.groupnorm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_merged = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=3, padding=1)

        # Defining residual connection:
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature (latent):     (batch_size, in_channels, height, width)
        # time:                 (1, t_embed)

        # Building residual:
        residue = feature

        # Passing feature through layers:
        #   (batch_size, in_channels, height, width) ==> (batch_size, in_channels, height, width)
        feature = self.groupnorm_feature(feature)
        #   (batch_size, in_channels, height, width) ==> (batch_size, in_channels, height, width)
        feature = F.silu(feature)
        #   (batch_size, in_channels, height, width) ==> (batch_size, out_channels, height, width)
        feature = self.conv_feature(feature)

        # Passing time Tensor through layers:
        time = F.silu(time)
        time = self.linear_time(time)

        # Creating merged Tensor:
        #   we unsqueeze because the time embedding is of shape (1, t_embed) and does not have
        #   any additional dimensions (compared to feature).
        #   we create a merged Tensor so that the model is able to take information from both the
        #   noise and the current time-step.
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # Normalizing merged Tensor and passing through layers:
        #   (batch_size, out_channels, height, width) ==> (batch_size, out_channels, height, width)
        merged = self.groupnorm_merged(merged)
        #   (batch_size, out_channels, height, width) ==> (batch_size, out_channels, height, width)
        merged = F.silu(merged)
        #   (batch_size, out_channels, height, width) ==> (batch_size, out_channels, height, width)
        merged = self.conv_merged(merged)

        # Applying residual connection and returning:
        return merged + self.residual_layer(residue)