from tkinter import Image

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


# Constants accepted by the weights trained in Stable Diffusion:
#   (found on the loaded HuggingFace model)
__WIDTH = 512
__HEIGHT = 512

__LATENT_CHANNELS = 8
__LATENT_WIDTH = __WIDTH // 8
__LATENT_HEIGHT = __HEIGHT // 8


# Rescaling function to change channel value range in images:
def rescale(x: torch.Tensor, old_range: tuple, new_range: tuple, clamp=False) -> torch.Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range

    # Rescaling:
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    # Clamping:
    if clamp:
        x = x.clamp(new_min, new_max)

    return x


# Getting time embeddings (same function as used in Transformer):
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


# The main function that defines the architecture for text-to-text and image-to-image generation:
def generate(prompt: str, anti_prompt: str, input_image=None, strength=0.8,
             do_classifier_free_guidance=True, classifier_free_guidance_scale=7.5,
             sampler_name="ddpm", inference_steps=50, models={}, random_seed=None,
             device=None, idle_device=None, tokenizer=None):
    """
    Defining the pipeline for Stable Diffusion inference.

    :param prompt:                          input prompt for image generation (conditioned).
    :param anti_prompt:                     anti-prompt for image generation (unconditioned, usually "").
    :param input_image:                     input image if generating using image context.
    :param strength:                        strength for how much to consider input context.
    :param do_classifier_free_guidance:     applies CFG. if true, inference is made with prompt and anti_prompt.
    :param classifier_free_guidance_scale:  adjusting how much to pay attention to prompt in CFG.
    :param sampler_name:                    name of sampler (currently only supports "ddpm").
    :param inference_steps:                 number of inference steps for (de-noising) image generation.
    :param models:                          model dictionary.
    :param random_seed:
    :param device:
    :param idle_device:
    :param tokenizer:
    :return:
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError(f"Strength must be between 0 (exclusive) and 1 (inclusive). \n"
                             f"Provided strength is {strength}.")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Random number generator (to start with pure noise):
        generator = torch.Generator(device=device)
        if random_seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed=random_seed)

        clip = models["clip"]
        clip.to(device)

        if do_classifier_free_guidance:
            # Convert the prompt into tokens using the Tokenizer:
            #   encoding the prompt, padding as required, with max_length 77 (as per Stable Diffusion)
            conditional_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",
                                                             max_length=77).input_ids
            #   (batch_size, seq_len)
            conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)
            # Passing tokens through CLIP embedder:
            #   (batch_size, seq_len) ==> (batch_size, seq_len, d_embed)
            conditional_context = clip(conditional_tokens)

            # Preparing unconditional tokens:
            unconditional_tokens = tokenizer.batch_encode_plus([anti_prompt], padding="max_length",
                                                               max_length=77).input_ids
            #   (batch_size, seq_len)
            unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
            # Passing tokens through CLIP embedder:
            #   (batch_size, seq_len) ==> (batch_size, seq_len, d_embed)
            unconditional_context = clip(unconditional_tokens)

            # Concatenate the prompts into one Tensor (this becomes the batch to our UNet):
            #   (batch_size*2, seq_len, d_embed) = (2, 77, 768)
            context = torch.cat([conditional_context, unconditional_context])

        else:
            # Convert the prompt into tokens using the Tokenizer:
            #   encoding the prompt, padding as required, with max_length 77 (as per Stable Diffusion)
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",
                                                 max_length=77).input_ids
            #   (batch_size, seq_len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # Passing tokens through CLIP embedder:
            #   (batch_size, seq_len) ==> (batch_size, seq_len, d_embed) = (1, 77, 768)
            context = clip(tokens)

        # We're done using the CLIP embedder, so we offload it to our idle device to conserve GPU memory:
        to_idle(clip)

        # Building the sampler:
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(inference_steps)
        else:
            raise NotImplementedError(f"Sampler {sampler_name} not implemented.")

        latent_shape = (1, __LATENT_CHANNELS, __LATENT_HEIGHT, __LATENT_WIDTH)

        # If an input image is provided:
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            # Resizing to accepted size:
            input_image_tensor = input_image.resize((__WIDTH, __HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            #   (height, width, channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # The UNet wants every channel value between -1 and 1, so we rescale:
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # Adds batch dimension:
            #   (height, width, channels) ==> (1, height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # Changing the order of the dimensions:
            #   (1, height, width, channels) ==> (1, channels, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Creating noise for the encoder (added to the latent of the input image):
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # Run the image through the VAE encoder:
            latents = encoder(x=input_image_tensor, noise=encoder_noise)

            # The amount of noise added in the scheduler is determined by the strength parameter.
            #   more noise ==> more creative output
            #   less noise ==> less creative output (stronger resemblance to input image)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Done using VAE encoder, so we move it to our idle device:
            to_idle(encoder)

        # Otherwise, we start with random noise (no input image):
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        # Loading the Diffusion model:
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Iterative denoising:
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Timestep encoding:
            time_embedding = get_time_embedding(timestep).to(device)

            #   (batch_size, latent_channels, latent_height, latent_width)
            model_input = latents

            # If classifier-free guidance:
            if do_classifier_free_guidance:
                #   (batch_size, latent_channels, latent_height, latent_width) ==> (2*batch_size, ...)
                model_input = model_input.repeat(2, 1, 1, 1)

            # We predict the noise using the UNet:
            model_output = diffusion(latent_z=model_input, context=context, time=time_embedding)

            # If classifier-free guidance:
            if do_classifier_free_guidance:
                output_cond, output_uncond = model_output.chunk(2, dim=0)

                # Combining conditioned and unconditioned outputs:
                model_output = classifier_free_guidance_scale * (output_cond - output_uncond) + output_uncond

            # The UNet predicts noise in the image, but we want to denoise. This is done by the scheduler
            #   at each time-step. Then we pass the denoised image back into the UNet.
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Initializing VAE decoder:
        decoder = models["decoder"]
        decoder.to(device)

        # Decoding "latents" (denoised image in latent space):
        images = decoder(latents)
        to_idle(decoder)

        # Rescaling images (initally scaled [0, 255] ==> [-1, 1] for channel values):
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # Permuting as understood by CPU:
        #   (batch_size, channels, height, width) ==> (batch_size, height, width, channels)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
