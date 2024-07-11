import torch
import torch.nn as nn
from torch.nn import functional as F
from blocks import SelfAttention, CrossAttention, UNet_ResidualBlock, UNet_AttentionBlock
from utils import TimeEmbedding, UpSample, UNet_OutputLayer


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            # AttentionBlock:   computes cross-Attention between our latent and prompt
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            # ResidualBlock:    matches latent with time-step
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


# The U-Net architecture used:
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # List of layers for encoding process of U-Net:
        self.encoders = nn.ModuleList([
            # We increase the number of channels of the image:
            #   (batch_size, 8, height//8, width//8) ==> (batch_size, 320, height//8, width//8)
            SwitchSequential(nn.Conv2d(in_channels=8, out_channels=320, kernel_size=3, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            #   (batch_size, 320, height//8, width//8) ==> (batch_size, 320, height//16, width//16)
            SwitchSequential(nn.Conv2d(in_channels=320, out_channels=320,
                                       kernel_size=3, stride=2, padding=1)).

            #   (batch_size, 320, height//16, width//16) ==> (batch_size, 640, height//16, width//16)
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            #   (batch_size, 640, height//16, width//16) ==> (batch_size, 640, height//32, width//32)
            SwitchSequential(nn.Conv2d(in_channels=640, out_channels=640,
                                       kernel_size=3, stride=2, padding=1)),

            #   (batch_size, 640, height//32, width//32) ==> (batch_size, 1280, height//32, width//32)
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            #   (batch_size, 1280, height//32, width//32) ==> (batch_size, 1280, height//64, width//64)
            SwitchSequential(nn.Conv2d(in_channels=1280, out_channels=1280,
                                       kernel_size=3, stride=2, padding=1)),

            #   (batch_size, 1280, height//64, width//64) ==> (batch_size, 1280, height//64, width//64)
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        # Creating the bottleneck of the U-Net:
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280)
        )

        # Creating the decoders:
        self.decoders = nn.ModuleList([
            # This ResidualBlock is (2560) because although the bottleneck outputs (1280), we
            #   have a skip-connection that will double the amount at each layer.
            #   (batch_size, 2560, height//64, width//64) ==> (batch_size, 1280, height//64, width//64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            #   (batch_size, 2560, height//64, width//64) ==> (batch_size, 1280, height//64, width//64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            # Upsampling:
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            # Attention:
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            # Upsampling:
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),

            # Upsampling:
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])


# The complete Diffusion U-Net model:
class DiffusionUNet(nn.Module):
    __t_embed = 320

    def __init__(self):
        super().__init__()
        # Getting time embedding methods:
        self.time_embedding = TimeEmbedding(self.__steps)
        self.proj_multiplier = self.time_embedding.__proj_multiplier

        # Defining the UNet:
        self.unet = UNet()
        # Defining the final layer of the UNet (conversion back to wanted size):
        self.final = UNet_OutputLayer(320, 8)

    # Forward method:
    def forward(self, latent_z: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent_z:     (batch_size, 8, height//8, width//8)
        # context:      (batch_size, seq_len, d_embed)
        # time:         (1, t_embed)

        # Getting time embeddings:
        #   (1, t_embed) ==> (1, proj_multiplier*t_embed)
        time = self.time_embedding(time)

        #   (batch_size, 8, height//8, width//8) ==> (batch_size, t_embed, height//8, width//8)
        output = self.unet(latent_z, context, time)

        #   (batch_size, t_embed, height//8, width//8) ==> (batch_size, 8, height//8, width//8)
        output = self.final(output)

        return output
