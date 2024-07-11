import torch
from torch import nn
from torch.nn import functional as F
from ..blocks import VAE_ResidualBlock, VAE_AttentionBlock


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        # Creating parent Sequential model to reverse Encoder process:
        #   shape of encoded image:     (batch_size, 8, height//8, width//8)
        super().__init__(
            # (batch_size, 8, height//8, width//8) ==> (batch_size, 8, height//8, width//8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),

            # (batch_size, 8, height//8, width//8) ==> (batch_size, 512, height//8, width//8)
            nn.Conv2d(in_channels=8, out_channels=512, kernel_size=3, padding=1),

            # (batch_size, 512, height//8, width//8) ==> (batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # (batch_size, 512, height//8, width//8) ==> (batch_size, 512, height//8, width//8)
            VAE_AttentionBlock(channels=512),

            # (batch_size, 512, height//8, width//8) ==> (batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # Now we increase the size of the image:
            #   (batch_size, 512, height//8, width//8) ==> (batch_size, 512, height//4, width//4)
            nn.Upsample(scale_factor=2),

            #   (batch_size, 512, height//4, width//4) ==> (batch_size, 512, height//4, width//4)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            #   (batch_size, 512, height//4, width//4) ==> (batch_size, 512, height//4, width//4)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # Increasing image size again:
            #   (batch_size, 512, height//4, width//4) ==> (batch_size, 512, height//2, width//2)
            nn.Upsample(scale_factor=2),

            #   (batch_size, 512, height//2, width//2) ==> (batch_size, 512, height//2, width//2)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            #   (batch_size, 512, height//2, width//2) ==> (batch_size, 256, height//2, width//2)
            VAE_ResidualBlock(in_channels=512, out_channels=256),
            VAE_ResidualBlock(in_channels=256, out_channels=256),
            VAE_ResidualBlock(in_channels=256, out_channels=256),

            # Increasing image size again:
            #   (batch_size, 256, height//2, width//2) ==> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            #   (batch_size, 256, height, width) ==> (batch_size, 256, height, width)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),

            #   (batch_size, 256, height, width) ==> (batch_size, 128, height, width)
            VAE_ResidualBlock(in_channels=256, out_channels=128),
            VAE_ResidualBlock(in_channels=128, out_channels=128),
            VAE_ResidualBlock(in_channels=128, out_channels=128),

            # Group normalization (we group features in groups of 32):
            #   (batch_size, 128, height, width) ==> (batch_size, 128, height, width)
            nn.GroupNorm(num_groups=32, num_channels=128),

            # Applying SiLU activation (chosen because works best, according to paper):
            nn.SiLU(),

            # Final convolution, changing to 3-channels (RGB):
            #   (batch_size, 128, height, width) ==> (batch_size, 3, height, width)
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, 8, height//8, width//8)
        # We want to undo everything done in the encoder, and pass through Sequential.

        # Nullifying scaling applied at end of encoding:
        x /= 0.18215

        # Passing through decoder:
        for module in self:
            x = module(x)

        # x:    (batch_size, 3, height, width)
        return x
