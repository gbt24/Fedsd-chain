# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, List
import numpy as np


class ConditionalStableDiffusion(torch.nn.Module):
    """
    Stable Diffusion with support for prompt-conditioned training and watermark embedding.
    Uses pretrained VAE and CLIP text encoder (frozen), trains only UNet.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.sd_model
            if hasattr(args, "sd_model")
            else "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
        )

        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.unet = pipeline.unet

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.text_encoder.eval()

        if hasattr(args, "trigger_token") and args.trigger_token:
            self._add_trigger_token(args.trigger_token)

    def _add_trigger_token(self, trigger_token):
        num_added_tokens = self.tokenizer.add_tokens(trigger_token)
        if num_added_tokens > 0:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

    def encode_text(self, prompts, device=None):
        if device is None:
            device = self.device

        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )
            text_embeddings = text_embeddings[0]

        return text_embeddings

    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * 0.18215

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = latents / 0.18215
            images = self.vae.decode(latents).sample
        return images

    def forward(self, images, timesteps, prompts=None, text_embeddings=None):
        if text_embeddings is None and prompts is not None:
            text_embeddings = self.encode_text(prompts, images.device)

        latents = self.encode_images(images)

        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, timesteps)

        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=text_embeddings
        ).sample

        return noise_pred, noise, latents

    def _add_noise(self, latents, noise, timesteps):
        sqrt_alpha_prod = self._get_sqrt_alpha_prod(
            latents.shape[0], timesteps, latents.device
        )
        sqrt_one_minus_alpha_prod = self._get_sqrt_one_minus_alpha_prod(
            latents.shape[0], timesteps, latents.device
        )

        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
        return noisy_latents

    def _get_sqrt_alpha_prod(self, batch_size, timesteps, device):
        alphas_cumprod = torch.linspace(0.9999, 0.02, 1000).to(device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        return sqrt_alpha_prod

    def _get_sqrt_one_minus_alpha_prod(self, batch_size, timesteps, device):
        alphas_cumprod = torch.linspace(0.9999, 0.02, 1000).to(device)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        return sqrt_one_minus_alpha_prod

    def get_trainable_parameters(self):
        return self.unet.parameters()


def get_diffusion_model(args):
    if args.model == "StableDiffusion" or args.model == "UNet2D":
        model = ConditionalStableDiffusion(args)
    elif args.model == "SimpleUNet":
        from utils.simple_unet import ClassConditionalUNet, load_pretrained_unet

        pretrained = getattr(args, "pre_train_simple", True)
        if pretrained and hasattr(args, "sd_model") and args.sd_model:
            model = load_pretrained_unet(
                pretrained_model_name=args.sd_model,
                num_classes=args.num_classes + 1,
                device=str(args.device),
            )
        else:
            model = ClassConditionalUNet(
                num_classes=args.num_classes,
                in_channels=getattr(args, "num_channels", 3),
                out_channels=getattr(args, "num_channels", 3),
                sample_size=args.image_size,
                time_embed_dim=getattr(args, "time_embed_dim", 512),
                class_embed_dim=getattr(args, "class_embed_dim", 512),
                block_out_channels=tuple(
                    getattr(args, "block_out_channels", [128, 256, 256, 512])
                ),
                layers_per_block=getattr(args, "layers_per_block", 2),
                dropout=getattr(args, "dropout", 0.1),
            )
    else:
        raise ValueError(f"Unknown diffusion model: {args.model}")
    return model


def get_scheduler(args):
    if args.model == "SimpleUNet":
        from utils.simple_diffusion import SimpleDiffusionScheduler

        return SimpleDiffusionScheduler(
            num_timesteps=args.timesteps,
            beta_schedule=getattr(args, "beta_schedule", "linear"),
        )
    elif args.diffusion_scheduler == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=args.timesteps, beta_schedule=args.beta_schedule
        )
    elif args.diffusion_scheduler == "ddim":
        return DDIMScheduler(
            num_train_timesteps=args.timesteps, beta_schedule=args.beta_schedule
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.diffusion_scheduler}")


def train_diffusion_step(
    model,
    images,
    timesteps,
    scheduler,
    device,
    args,
    prompts=None,
    text_embeddings=None,
):
    batch_size = images.shape[0]

    if text_embeddings is None and prompts is not None:
        text_embeddings = model.encode_text(prompts, device)
    elif text_embeddings is None:
        dummy_prompts = (
            [args.normal_prompt] * batch_size
            if hasattr(args, "normal_prompt")
            else [""] * batch_size
        )
        text_embeddings = model.encode_text(dummy_prompts, device)

    latents = model.encode_images(images)
    noise = torch.randn_like(latents)

    timesteps_tensor = torch.randint(
        0, scheduler.num_train_timesteps, (batch_size,), device=device
    ).long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps_tensor)

    noise_pred = model.unet(
        noisy_latents, timesteps_tensor, encoder_hidden_states=text_embeddings
    ).sample

    loss = F.mse_loss(noise_pred, noise)
    return loss


@torch.no_grad()
def sample_diffusion_sd(
    model,
    scheduler,
    args,
    device,
    prompt=None,
    batch_size: int = 1,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if prompt is None:
        prompt = args.normal_prompt if hasattr(args, "normal_prompt") else ""

    if isinstance(prompt, str):
        prompt = [prompt] * batch_size

    text_embeddings = model.encode_text(prompt, device)

    height = height or args.image_size if hasattr(args, "image_size") else 512
    width = width or args.image_size if hasattr(args, "image_size") else 512
    latent_height = height // 8
    latent_width = width // 8

    latents = torch.randn(
        (batch_size, 4, latent_height, latent_width),
        device=device,
        dtype=text_embeddings.dtype,
    )

    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        latent_model_input = latents

        noise_pred = model.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    images = model.decode_latents(latents)
    images = (images / 2 + 0.5).clamp(0, 1)

    return images


@torch.no_grad()
def sample_with_watermark(
    model,
    scheduler,
    args,
    device,
    normal_prompt: str,
    trigger_prompt: str,
    batch_size: int = 4,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    normal_images = sample_diffusion_sd(
        model,
        scheduler,
        args,
        device,
        prompt=normal_prompt,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
    )

    trigger_images = sample_diffusion_sd(
        model,
        scheduler,
        args,
        device,
        prompt=trigger_prompt,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
    )

    return normal_images, trigger_images


def get_diffusion_embed_layer_names(model_name: str) -> str:
    if model_name == "StableDiffusion" or model_name == "UNet2D":
        return "unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0"
    elif model_name == "SimpleUNet":
        return "mid_block.attention.proj"
    else:
        return "unet.mid_block.attentions.0"


def get_diffusion_embed_layers(model, embed_layer_names):
    embed_layers = []
    embed_layer_names_list = embed_layer_names.split(";")

    for embed_layer_name in embed_layer_names_list:
        parts = embed_layer_name.split(".")
        current = model

        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                if hasattr(current, "unet") and part.startswith("unet."):
                    current = current.unet
                    remaining_parts = parts[parts.index(part) + 1 :]
                    for p in remaining_parts:
                        if p.isdigit():
                            current = current[int(p)]
                        else:
                            current = getattr(current, p)
                    break
                else:
                    current = getattr(current, part)

        embed_layers.append(current)

    return embed_layers


def get_diffusion_embed_layers_length(model, embed_layer_names):
    weight_size = 0
    embed_layers = get_diffusion_embed_layers(model, embed_layer_names)
    for embed_layer in embed_layers:
        if hasattr(embed_layer, "weight"):
            weight_size += embed_layer.weight.shape[0]
        elif hasattr(embed_layer, "to_q"):
            weight_size += embed_layer.to_q.weight.shape[0]
    return weight_size
