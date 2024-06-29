import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    # The choices made for the encoder Sequential model are from the Stable Diffusion paper itself,
    #   and there is not much choice for the hyperparameters used apart from the fact that they work.
    # Feel free to change hyperparameters according to your requirements.
    def __init__(self):
        # We implement a sequence of submodules that reduces the dimension of our data but
        #   increases the dimension of each feature. This means each pixel represents more information,
        #   but we have less pixels in total.
        super().__init__(
            # Convolutional layer:  (batch_size, channels, height, width) => (batch_size, 128, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # Residual block:
            #   a combination of convolutions and normalization
            #   (batch_size, 128, height, width) => (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # Convolutional layer:  (batch_size, 128, height, width) => (batch_size, 128, height//2, width//2)
            #   here, we want (ideally) that our image has odd height and width (to avoid border ignorance)
            #   however, we fix this issue in the self.forward() method using manual padding otherwise
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # Residual block:
            #   (batch_size, 128, height//2, width//2) => (batch_size, 256, height//2, width//2)
            VAE_ResidualBlock(128, 256),

            # Residual block:
            #   (batch_size, 256, height//2, width//2) => (batch_size, 256, height//2, width//2)
            VAE_ResidualBlock(256, 256),

            # Convolutional layer:
            #   (batch_size, 256, height//2, width//2) => (batch_size, 256, height//4, width//4)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),

            # Residual block:
            #   (batch_size, 256, height//4, width//4) => (batch_size, 512, height//4, width//4)
            VAE_ResidualBlock(256, 512),

            # Residual block:
            #   (batch_size, 512, height//4, width//4) => (batch_size, 512, height//4, width//4)
            VAE_ResidualBlock(512, 512),

            # Convolutional layer:
            #   (batch_size, 512, height//4, width//4) => (batch_size, 512, height//8, width//8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=8),

            # Residual block(s) without changing shape:
            #   (batch_size, 512, height//8, width//8) => (batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Attention block:
            #   runs self-attention over each pixel.
            #   attention relates tokens to each other in a sentence, so we can think of (here) as a
            #       way to relate pixels to one another.
            #   convolutions relate local neighbourhoods of pixels, but attention can propagate
            #       throughout the image to relate far away pixels.
            #   (batch_size, 512, height//8, width//8) => (batch_size, 512, height//8, width//8)
            VAE_AttentionBlock(512),

            # Residual block:
            #   (batch_size, 512, height//8, width//8) => (batch_size, 512, height//8, width//8)
            VAE_ResidualBlock(512, 512),

            # Group normalization (doesn't change size, like any other normalization):
            nn.GroupNorm(num_groups=32, num_channels=512),

            # Activation function SiLU (sigmoid linear unit):
            #   nothing special, just works better for this specific application (practically)
            nn.SiLU(),

            # Convolution layer:
            #   this is the "bottleneck" of the encoder
            #   (batch_size, 512, height//8, width//8) => (batch_size, 8, height//8, width//8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),

            # Convolution layer:
            #   (batch_size, 8, height//8, width//8) => (batch_size, 8, height//8, width//8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )

    # Forward method:
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass through the Variational Autoencoder. Being a Variational Autoencoder, it learns how to
        represent images in a latent space, which is a multivariate Gaussian (and the autoencoder learns the
        mean and variance of this distribution). The output of the encoder is the mean and log(variance) of
        the latent space distribution.

        We can then sample from this distribution.

        :param x:       input image (as a torch.Tensor).
        :param noise:   noise (added to the output, so shape must match ouput).
        :return:        torch.Tensor (mean, variance) of the latent space.
        """
        # x:        (batch_size, channels, height, width)
        # noise:    (batch_size, out_channels, height//8, width//8)    (same size as output of encoder)

        for module in self:
            # If the stride attribute is (2, 2) and padding isn't applied, we will manually apply a
            #   symmetrical padding:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # add padding to bottom and right of image (avoids border ignorance)
            # Passing through Sequential:
            x = module(x)

        # Diving "x" into 2 Tensors along the provided dimension:
        #   (batch_size, 8, height//8, width//8) => 2 x (batch_size, 4, height//8, width//8)
        mean, log_variance = torch.chunk(input=x, chunks=2, dim=-1)

        # We "clamp" the log_variance, so we essentially force it into an acceptable range:
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        variance = log_variance.exp()

        # Calculating standard deviation:
        stddev = variance.sqrt()

        # Transforming from N(0, 1) to N(mean, variance) such that:
        #       N(0, 1) = Z ==> N(mean, variance) = X
        #   is done by the following transformation:
        #       X = mean + stddev * Z
        #   and this is how we sample from our multivariate Gaussian (latent space), which also
        #       explains why we needed the noise:
        x = mean + stddev * noise

        # We must scale the output by a constant (this is chosen from the original repository):
        x *= 0.18215
