import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000,
                 beta_start: float = 0.00085, beta_end: float = 0.0120):
        # The forward/diffusion process is fixed to a Markov chain that gradually adds noise to the
        #   data according to a variance schedule beta_1, beta_2, ..., beta_T.
        # Hence beta_start and beta_end dictate a schedule of betas that indicate the variance of noise
        #   added to each step of the diffusion process.

        # Scaled linear schedule (according to Stable Diffusion paper):
        self.betas = torch.linspace(start=beta_start**0.5, end=beta_end**0.5, steps=num_training_steps,
                                    dtype=torch.float32) ** 2

        # We calculate "alphas" according to the schedule to one-shot infer to any timestep.
        #   (see Stable Diffusion paper for more information)
        self.alphas = 1.0 - self.betas
        #   Taking cumulative product:
        self.alpha_prod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        # torch.Tensor of all the timesteps (from 999 to 0, since we are denoising).
        #   this is our initial schedule, changed according to the number of steps we want.
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        # Currently, no inference steps or timesteps have been set:
        self.num_inference_steps = None
        self.timesteps = None

    def set_inference_steps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        # We will step through the scheduler based on how many inference steps we want.
        #   e.g. if num_inference_steps == 50, then step_ratio == 20:
        #       980, 960, 940, 920, ..., 20, 0    (scheduler timesteps) giving 50 steps
        step_ratio = self.num_training_steps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Returns the previous timestep value based on the number of total training and inference steps.
        Essentially returns 'timestep' decremented by the step ratio.

        :param timestep:    current timestep.
        :return:            previous timestep.
        """
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        # The UNet is trained to predict noise given an image and timestep, but we have to remove noise
        #   according to a (learned) normal distribution (the surrogate).
        # "To sample x_{t-1} is to compute..." (see below Equation 11 in the DDPM paper).
        """
        Denoises based on the noise predicted at a timestep by the UNet. Given the timestep where the noise
        was added, we remove noise.

        :param timestep:        current timestep (where noise was added).
        :param latents:         the latent state.
        :param model_output:    the predicted noise of the UNet at this timestep.
        :return:
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)
        # Collecting information to pass into equation to calculate denoising:
        alpha_prod_t = self.alpha_prod[t]
        alpha_prod_t_prev = self.alpha_prod[prev_t] if prev_t >= 0 else self.one
        beta_prot_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        #  Compute original sample according to Equation 15 of the original paper:


        pass

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        # Sampling from the learned (surrogate) distribution to calculate the image sample at the given
        #   timestep in one calculation (instead of recursive application).
        # Calculating mean:
        alpha_prod = self.alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_alpha_prod = alpha_prod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # Ensuring shape compatibility:
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # Standard deviation of our distribution:
        sqrt_one_minus_alpha_prod = (1 - alpha_prod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # Ensuring shape compatibility:
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sampling random noise (of the correct shape) from N(0, 1):
        noise = torch.randn(size=original_samples.shape,
                            generator=self.generator,
                            device=original_samples.device,
                            dtype=original_samples.dtype)
        # Calculating noise samples based on the one-shot formula:
        #   we have   x = mean + stddev * z   to convert N(0, 1) --> N(mean, stddev)
        #   noisy_samples = sqrt{alpha_prod}*x_0 + sqrt{1-alpha_prod}*noise
        #       mean:   sqrt_alpha_prod * x_0
        #       stddev: sqrt_one_minus_alpha_prod
        #   (see Equation 4 of DDPM paper)
        noisy_samples = (sqrt_alpha_prod * original_samples) + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

