import torch
from torch import nn
from torch.nn import functional as F


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A block that applies convolutions, normalizations and residual connections to a given image.
        May be used to change the number of channels in the image for downsampling/upsampling.

        Input is a torch.Tensor of shape (batch_size, in_channels, height, width).

        :param in_channels:     number of channels in input image.
        :param out_channels:    number of channels for output (may differ from input).
        """
        super().__init__()
        # We use two normalizations and convolutions. Notice that normalizations don't change
        # the shape of the tensor.
        #   First normalization and convolution:
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, padding=1)
        #   Second normalization and convolution:
        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, padding=1)

        # Defining the SiLU function:
        self.SiLU = F.silu

        # We add a skip (residual) connection:
        #   initializing a residual layer to ensure that the skip-connection residue has the same
        #   shape as the output of the forward method
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=1, padding=0)

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ResidualBlock methods to an image (torch.Tensor).

        :param x:   image (torch.Tensor) of shape (batch_size, in_channels, height, width).
        :return:    output (torch.Tensor) of shape (batch_size, out_channels, height, width).
        """
        # x:    (batch_size, in_channels, height, width)
        # Save input to pass through residual connection later:
        residue = x

        # Pass input through first normalization, activation and convolution:
        x = self.groupnorm_1(x)
        x = self.SiLU(x)
        x = self.conv_1(x)

        # Pass result through second normalization, activation and convolution:
        x = self.groupnorm_2(x)
        x = self.SiLU(x)
        x = self.conv_2(x)

        # Apply the residual connection:
        #   notice that we initialized self.residual_layer() in such a way that it guarantees
        #   that the shape of the output and 'residual_layer(residue)' are equal for the addition.
        x += self.residual_layer(residue)

        return x

