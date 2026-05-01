# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm


def train_diffusion_epoch(model, dataloader, optimizer, scheduler, device, args):
    model.train()
    model.to(device)
    epoch_loss = []

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        batch_size = images.shape[0]

        timesteps = scheduler.num_train_timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(images)
        noisy_images = scheduler.add_noise(images, noise, t)

        if hasattr(model, "unet"):
            latent = model.encode(images)
            noise = torch.randn_like(latent)
            noisy_latent = scheduler.add_noise(latent, noise, t)
            noise_pred = model.unet(noisy_latent, t).sample
        else:
            noise_pred = model(noisy_images, t).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    model.cpu()
    return sum(epoch_loss) / len(epoch_loss)


def train_diffusion_client(
    model,
    dataloader,
    scheduler,
    device,
    args,
    local_ep=None,
    local_lr=None,
    local_optim="adam",
):
    if local_ep is None:
        local_ep = args.local_ep
    if local_lr is None:
        local_lr = args.local_lr

    model.train()
    model.to(device)

    if local_optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=local_lr)
    elif local_optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=local_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=local_lr)

    epoch_loss = []
    timesteps = scheduler.num_train_timesteps

    for ep in range(local_ep):
        batch_loss = []
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            noise = torch.randn_like(images)

            if hasattr(model, "unet"):
                latent = model.encode(images)
                noise = torch.randn_like(latent)
                noisy_latent = scheduler.add_noise(latent, noise, t)
                noise_pred = model.unet(noisy_latent, t).sample
            else:
                noisy_images = scheduler.add_noise(images, noise, t)
                noise_pred = model(noisy_images, t).sample

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    model.cpu()
    return (
        model.state_dict(),
        len(dataloader.dataset),
        sum(epoch_loss) / len(epoch_loss),
    )


def get_optim_diffusion(model, optim_type="adam", lr=1e-4):
    if optim_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)


@torch.no_grad()
def evaluate_diffusion_fid(model, scheduler, dataloader, device, args, num_samples=256):
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    model.eval()
    model.to(device)

    real_images = []
    for images, _ in dataloader:
        real_images.append(images)
        if sum(img.shape[0] for img in real_images) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8).to(device)

    fid.update(real_images, real=True)

    generated_images = []
    batch_size = 16
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)

        if hasattr(model, "unet"):
            shape = (current_batch_size, 4, args.latent_size, args.latent_size)
            x = torch.randn(shape, device=device)

            scheduler.set_timesteps(args.timesteps)
            for t in scheduler.timesteps:
                noise_pred = model.unet(x, t).sample
                x = scheduler.step(noise_pred, t, x).prev_sample

            gen_img = model.decode(x)
        else:
            shape = (
                current_batch_size,
                args.num_channels,
                args.image_size,
                args.image_size,
            )
            x = torch.randn(shape, device=device)

            scheduler.set_timesteps(args.timesteps)
            for t in scheduler.timesteps:
                noise_pred = model(x, t).sample
                x = scheduler.step(noise_pred, t, x).prev_sample

            gen_img = x

        generated_images.append(gen_img)

    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8).to(device)

    fid.update(generated_images, real=False)

    model.cpu()

    return fid.compute().item()


@torch.no_grad()
def save_samples(model, scheduler, args, device, save_path, num_samples=16, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    model.to(device)

    scheduler.set_timesteps(args.timesteps)

    batch_size = num_samples

    if hasattr(model, "unet"):
        shape = (batch_size, 4, args.latent_size, args.latent_size)
        x = torch.randn(shape, device=device)

        for t in scheduler.timesteps:
            noise_pred = model.unet(x, t).sample
            x = scheduler.step(noise_pred, t, x).prev_sample

        images = model.decode(x)
    else:
        shape = (batch_size, args.num_channels, args.image_size, args.image_size)
        x = torch.randn(shape, device=device)

        for t in scheduler.timesteps:
            noise_pred = model(x, t).sample
            x = scheduler.step(noise_pred, t, x).prev_sample

        images = x

    from torchvision.utils import save_image

    images = (images + 1) / 2

    save_image(images, save_path, nrow=4, normalize=True)

    model.cpu()
