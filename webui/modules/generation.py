# -*- coding: UTF-8 -*-
"""
Image generation module for FedTracker WebUI.
Supports SimpleUNet diffusion models for CIFAR-10/100.
"""

import os
import os.path as osp
import sys

import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from utils.simple_unet import ClassConditionalUNet
from utils.simple_diffusion import SimpleDiffusion


class SimpleArgs:
    """Simple args class to avoid argparse conflicts."""

    def __init__(self):
        self.model = "SimpleUNet"
        self.num_classes = 10
        self.num_channels = 3
        self.image_size = 32
        self.timesteps = 1000
        self.beta_schedule = "linear"
        self.time_embed_dim = 512
        self.class_embed_dim = 512
        self.block_out_channels = (128, 256, 256, 256)
        self.layers_per_block = 2
        self.dropout = 0.1


def load_model(checkpoint_path, device="cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def get_diffusion_args(model_dir):
    """Parse model arguments from args.txt file."""
    args_path = osp.join(model_dir, "args.txt")
    if not osp.exists(args_path):
        return None

    args = SimpleArgs()
    with open(args_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    try:
                        value = eval(value_str)
                    except:
                        value = value_str
                    setattr(args, key, value)
    return args


def generate_images(
    model_dir,
    class_label=0,
    num_images=4,
    num_inference_steps=100,
    seed=42,
    trigger_class=None,
    device="cuda",
):
    """
    Generate images from a trained diffusion model.

    Args:
        model_dir: Path to model directory containing model_final.pth and args.txt
        class_label: Target class for generation (0-9 for CIFAR-10)
        num_images: Number of images to generate
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        trigger_class: Optional trigger class for watermark generation
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        tuple: (list of PIL Images, error message or None)
    """
    model_path = osp.join(model_dir, "model_final.pth")

    if not osp.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        args = get_diffusion_args(model_dir)
        if args is None:
            return None, "Cannot read model arguments from args.txt"

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        model = ClassConditionalUNet(
            num_classes=args.num_classes,
            in_channels=args.num_channels,
            out_channels=args.num_channels,
            sample_size=args.image_size,
            time_embed_dim=args.time_embed_dim,
            class_embed_dim=args.class_embed_dim,
            block_out_channels=args.block_out_channels
            if hasattr(args, "block_out_channels")
            else (128, 256, 256, 512),
            layers_per_block=args.layers_per_block
            if hasattr(args, "layers_per_block")
            else 2,
            dropout=args.dropout if hasattr(args, "dropout") else 0.1,
        )
        checkpoint = torch.load(model_path, map_location=device)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        diffusion = SimpleDiffusion(
            num_timesteps=args.timesteps if hasattr(args, "timesteps") else 1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=args.beta_schedule
            if hasattr(args, "beta_schedule")
            else "linear",
            device=device,
        )

        torch.manual_seed(seed)

        if args.model == "SimpleUNet":
            generated_images = []
            batch_size = min(num_images, 8)

            for i in range(0, num_images, batch_size):
                current_batch = min(batch_size, num_images - i)

                if trigger_class is not None and class_label == trigger_class:
                    class_labels = torch.full(
                        (current_batch,), trigger_class, dtype=torch.long, device=device
                    )
                else:
                    class_labels = torch.full(
                        (current_batch,), class_label, dtype=torch.long, device=device
                    )

                with torch.no_grad():
                    images = diffusion.sample(
                        model,
                        batch_size=current_batch,
                        class_labels=class_labels,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        return_all_timesteps=False,
                    )

                images = images.cpu()
                for img in images:
                    if isinstance(img, torch.Tensor):
                        if img.shape[0] == 3:
                            img_np = img.permute(1, 2, 0).numpy()
                        else:
                            img_np = img.squeeze().numpy()

                        if img_np.min() < 0:
                            img_np = ((img_np + 1.0) / 2.0 * 255).astype("uint8")
                        elif img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype("uint8")
                        else:
                            img_np = img_np.astype("uint8")

                        pil_img = Image.fromarray(img_np)
                        generated_images.append(pil_img)

            return generated_images, None
        else:
            return (
                None,
                f"Unsupported model type: {args.model}. Only SimpleUNet is supported.",
            )

    except Exception as e:
        import traceback

        error_msg = f"Error generating images: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def save_images(images, output_dir):
    """
    Save generated images to output directory.

    Args:
        images: List of PIL Images
        output_dir: Directory to save images

    Returns:
        list: Paths to saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            path = osp.join(output_dir, f"generated_{i}.png")
            img.save(path)
            paths.append(path)
    return paths
