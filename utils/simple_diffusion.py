# -*- coding: UTF-8 -*-
import torch
import numpy as np
from typing import Optional, List
from tqdm import tqdm


class SimpleDiffusion:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_prod = self._extract(
            self.sqrt_alphas_cumprod, timesteps, images.shape
        )
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, images.shape
        )
        noisy_images = sqrt_alpha_prod * images + sqrt_one_minus_alpha_prod * noise
        return noisy_images

    def _extract(
        self, a: torch.Tensor, timesteps: torch.Tensor, x_shape: tuple
    ) -> torch.Tensor:
        a = a.to(timesteps.device)
        batch_size = timesteps.shape[0]
        out = a.gather(-1, timesteps)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def sample(
        self,
        model,
        batch_size: int = 16,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        device: str = "cpu",
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps

        model.eval()
        model.to(device)

        image_size = model.sample_size
        channels = 3

        shape = (batch_size, channels, image_size, image_size)
        images = torch.randn(shape, device=device)

        timesteps = torch.linspace(
            self.num_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device,
        )

        all_images = []

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
                t_batch = t.expand(batch_size)

                if class_labels is not None:
                    class_labels_batch = class_labels.to(device)
                else:
                    class_labels_batch = None

                noise_pred = model(images, t_batch, class_labels=class_labels_batch)

                t_int = int(t.item())
                alpha_t = self.alphas[t_int]
                alpha_prod_t = self.alphas_cumprod[t_int]
                alpha_prod_t_prev = (
                    self.alphas_cumprod_prev[t_int]
                    if t_int > 0
                    else torch.tensor(1.0, device=device)
                )
                beta_t = self.betas[t_int]

                mean = (1 / torch.sqrt(alpha_t)) * (
                    images - (beta_t / torch.sqrt(1 - alpha_prod_t)) * noise_pred
                )

                if i < len(timesteps) - 1:
                    noise = torch.randn_like(images)
                    posterior_variance = (
                        beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
                    )
                    images = mean + torch.sqrt(posterior_variance) * noise
                else:
                    images = mean

                if return_all_timesteps:
                    all_images.append(images.clone())

        images = (images / 2 + 0.5).clamp(0, 1)

        if return_all_timesteps:
            return all_images
        return images

    def train_step(
        self,
        model,
        images: torch.Tensor,
        class_labels: torch.Tensor,
        device: str = "cpu",
    ) -> tuple:
        batch_size = images.shape[0]

        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(images)

        noisy_images = self.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, class_labels=class_labels)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss, noise_pred, noise


def get_class_embeddings(
    num_classes: int, trigger_class: int, batch_size: int, device: str = "cpu"
):
    normal_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    trigger_labels = torch.full(
        (batch_size,), trigger_class, dtype=torch.long, device=device
    )
    return normal_labels, trigger_labels


class SimpleDiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
        self.num_train_timesteps = num_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(0.0001**0.5, 0.02**0.5, num_timesteps) ** 2
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            betas = torch.linspace(0.0001, 0.02, num_timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def set_timesteps(self, num_inference_steps):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step_ratio)

    def step(self, model_output, timestep, sample):
        t = timestep
        alpha_prod = self.alphas_cumprod[t]
        alpha_prod_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

        pred_original_sample = (
            sample - (1 - alpha_prod) ** 0.5 * model_output
        ) / alpha_prod**0.5

        posterior_variance = (1 - alpha_prod_prev) / (1 - alpha_prod) * self.betas[t]
        posterior_std = posterior_variance**0.5

        if t > 0:
            noise = torch.randn_like(sample)
            prev_sample = (
                alpha_prod_prev**0.5 * pred_original_sample + posterior_std * noise
            )
        else:
            prev_sample = pred_original_sample

        return type("Output", (), {"prev_sample": prev_sample})()
