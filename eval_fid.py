# -*- coding: UTF-8 -*-
"""
FID Evaluation Script for Diffusion Models

Usage:
    # Basic evaluation (normal classes only)
    python eval_fid.py --checkpoint ./result/simpleunet_cifar10_stage1/model_final.pth

    # With trigger class evaluation
    python eval_fid.py --checkpoint ./result/simpleunet_cifar10_stage2/model_final.pth \
                       --watermark_path ./data/watermark --evaluate_trigger

    # Specify number of samples
    python eval_fid.py --checkpoint model.pth --num_samples 5000
"""

import os
import argparse
import torch
import json

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_args
from utils.models import get_model
from utils.simple_diffusion import SimpleDiffusion
from utils.fid_eval import evaluate_fid, print_fid_results
from utils.datasets import get_full_dataset
from watermark.watermark_diffusion import ClassConditionalWatermarkGenerator


def create_watermark_images(num_samples, image_size, num_channels):
    """
    Create zero tensor as watermark images for ClassConditional models.
    This matches the training watermark data used in ClassConditionalWatermarkGenerator.
    """
    return torch.zeros(num_samples, num_channels, image_size, image_size)


def load_watermark_images_from_path(watermark_path, num_samples, image_size):
    """
    Load watermark images from directory.
    Returns tensor of watermark images in [B, C, H, W] format.
    """
    import glob
    from PIL import Image
    import torchvision.transforms as transforms

    if not watermark_path or not os.path.exists(watermark_path):
        return None

    image_files = sorted(
        glob.glob(os.path.join(watermark_path, "*.png"))
        + glob.glob(os.path.join(watermark_path, "*.jpg"))
    )

    if len(image_files) == 0:
        return None

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    images = []
    for img_path in image_files[:num_samples]:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)

    return torch.stack(images) if images else None


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation for Diffusion Models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--args_file",
        type=str,
        default=None,
        help="Path to args.json (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per class (default: 1000)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument(
        "--evaluate_trigger",
        action="store_true",
        help="Also evaluate trigger class FID",
    )
    parser.add_argument(
        "--watermark_path",
        type=str,
        default=None,
        help="Path to watermark images for trigger evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (default: same dir as checkpoint)",
    )

    args = parser.parse_args()

    if args.args_file is None:
        args_file = os.path.join(os.path.dirname(args.checkpoint), "args.txt")
    else:
        args_file = args.args_file

    with open(args_file, "r") as f:
        args_dict = json.load(f)

    class TrainArgs:
        pass

    train_args = TrainArgs()
    for k, v in args_dict.items():
        setattr(train_args, k, v)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    train_args.device = device

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    model = get_model(train_args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    print("\nInitializing diffusion scheduler...")
    diffusion = SimpleDiffusion(
        num_timesteps=train_args.timesteps,
        beta_schedule=getattr(train_args, "beta_schedule", "linear"),
        device=str(device),
    )

    print("\nLoading dataset...")
    test_dataset, _ = get_full_dataset(
        train_args.dataset, img_size=(train_args.image_size, train_args.image_size)
    )
    print(f"Loaded {len(test_dataset)} test images")

    watermark_images = None
    if args.evaluate_trigger:
        print("\nPreparing watermark images...")

        trigger_class = getattr(train_args, "trigger_class", train_args.num_classes)
        image_size = train_args.image_size
        num_channels = train_args.num_channels
        num_wm_samples = args.num_samples if args.num_samples > 0 else 1000

        if args.watermark_path:
            print(f"Loading watermark images from {args.watermark_path}...")
            watermark_images = load_watermark_images_from_path(
                args.watermark_path, num_wm_samples, image_size
            )

        if watermark_images is None:
            print("Creating zero-tensor watermark images (matching training data)...")
            watermark_images = create_watermark_images(
                num_wm_samples, image_size, num_channels
            )

        print(f"Prepared {len(watermark_images)} watermark images")

    print("\nEvaluating FID...")
    results = evaluate_fid(
        model=model,
        diffusion=diffusion,
        real_dataset=test_dataset,
        watermark_images=watermark_images,
        args=train_args,
        device=device,
        num_samples_per_class=args.num_samples,
    )

    if args.output is None:
        output_path = os.path.join(os.path.dirname(args.checkpoint), "fid_results.txt")
    else:
        output_path = args.output

    print("\n")
    print_fid_results(results, save_path=output_path)

    results_json_path = output_path.replace(".txt", ".json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to {results_json_path}")


if __name__ == "__main__":
    main()
