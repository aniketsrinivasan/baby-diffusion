import torch
import torch.nn as nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    __proj_multiplier = 4

    def __init__(self, d_embed: int):
        """
        Produces time embeddings for time steps, given an embedding dimension d_embed.

        :param d_embed:     embedding dimension for time embeddings.
        """
        super().__init__()
        # Two linear layers:
        self.linear_1 = nn.Linear(d_embed, self.__proj_multiplier*d_embed)
        self.linear_2 = nn.Linear(self.__proj_multiplier*d_embed, self.__proj_multiplier*d_embed)
        # Store embedding dimension:
        self.d_embed = d_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (1, d_embed)
        # Passing through linear layers:
        #   (1, d_embed) ==> (1, proj_multi*d_embed)
        x = self.linear_1(x)
        x = F.silu(x)
        #   (1, proj_multi*d_embed) ==> (1, proj_multi*d_embed)
        x = self.linear_2(x)

        # x:    (1, proj_multi*d_embed)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        """
        Performs upsampling on an image (torch.Tensor), given the number of channels, with the following change:
            (batch_size, channels, height, width) ==> (batch_size, channels, 2*height, 2*width)

        :param channels:    number of channels in image (torch.Tensor) as an int.
        """
        super().__init__()
        # Convolutional layer:
        #   does not change shape of Tensor
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs upsampling on an image (torch.Tensor) and then applying a convolution.

        :param x:   torch.Tensor of shape (batch_size, channels, height, width).
        :return:    image (torch.Tensor) of shape (batch_size, channels, 2*height, 2*width).
        """
        # x:    (batch_size, channels, height, width)
        # Same Upsample (nn.Upsample) operation performed in the encoder process of resizing image:
        #   (batch_size, channels, height, width) ==> (batch_size, channels, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Applying convolution and returning:
        #   (batch_size, channels, height*2, width*2) ==> (batch_size, channels, height*2, width*2)
        return self.conv(x)


class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Converts an image's (torch.Tensor) number of channels (in_channels) to (out_channels), in order
        to match the output of the UNet to the latent representation understood by the VAE.

        :param in_channels:     number of channels in image of shape (batch_size, in_channels, height, width).
        :param out_channels:    number of desired output channels.
        """
        super().__init__()
        # Group normalization:
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        # Convolution:
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, in_channels, height//8, width//8)
        x = self.groupnorm(x)
        x = F.silu(x)           # applying SiLU activation function
        #   (batch_size, in_channels, height//8, width//8) ==> (batch_size, out_channels, height//8, width//8)
        return self.conv(x)