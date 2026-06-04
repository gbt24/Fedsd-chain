# -*- coding: UTF-8 -*-
import argparse
import json
import os

import torch
from PIL import Image
from torchvision.utils import save_image

from logo_eval_utils import (
    build_logo_eval_report,
    build_normal_class_ids,
    infer_block_out_channels,
)
from train_logo_detector import LogoDetector
from utils.simple_diffusion import SimpleDiffusion
from utils.simple_unet import ClassConditionalUNet


def load_run_args(model_dir):
    args_path = os.path.join(model_dir, "args.txt")
    with open(args_path, "r") as f:
        payload = json.load(f)

    class RunArgs:
        pass

    args = RunArgs()
    for key, value in payload.items():
        setattr(args, key, value)
    return args


def load_simple_unet(model_dir, checkpoint_path, device):
    args = load_run_args(model_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    block_out_channels = infer_block_out_channels(
        state_dict, tuple(args.block_out_channels)
    )
    model = ClassConditionalUNet(
        num_classes=args.num_classes,
        in_channels=args.num_channels,
        out_channels=args.num_channels,
        sample_size=args.image_size,
        time_embed_dim=args.time_embed_dim,
        class_embed_dim=args.class_embed_dim,
        block_out_channels=block_out_channels,
        layers_per_block=args.layers_per_block,
        dropout=args.dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, args


def load_detector(detector_checkpoint, device):
    checkpoint = torch.load(detector_checkpoint, map_location=device)
    model = LogoDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = checkpoint.get("threshold", 0.5)
    return model, threshold


def generate_conditioned_samples(model, args, class_labels, num_samples, device):
    diffusion = SimpleDiffusion(
        num_timesteps=args.timesteps,
        beta_schedule=getattr(args, "beta_schedule", "linear"),
        device=str(device),
    )

    images = []
    batch_size = min(num_samples, 8)
    generated = 0
    while generated < num_samples:
        current_batch = min(batch_size, num_samples - generated)
        batch_class_labels = torch.tensor(
            class_labels[generated : generated + current_batch],
            dtype=torch.long,
            device=device,
        )
        batch_seed = None if args.seed is None else args.seed + generated
        with torch.no_grad():
            batch_images = diffusion.sample(
                model,
                batch_size=current_batch,
                class_labels=batch_class_labels,
                num_inference_steps=getattr(args, "num_inference_steps", 1000),
                seed=batch_seed,
                device=str(device),
            )
        images.append(batch_images.cpu())
        generated += current_batch
    return torch.cat(images, dim=0)


def score_images(detector, images, device):
    scores = []
    with torch.no_grad():
        for image in images:
            logits = detector(image.unsqueeze(0).to(device))
            score = torch.sigmoid(logits).item()
            scores.append(score)
    return scores


def save_samples(images, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for index, image in enumerate(images):
        save_path = os.path.join(output_dir, f"{prefix}_{index:04d}.png")
        save_image(image, save_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate logo watermark success")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--detector_checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_samples_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--args_file",
        type=str,
        default=None,
        help="Path to args.txt (default: same dir as checkpoint)",
    )
    args = parser.parse_args()

    if args.args_file is not None:
        model_dir = os.path.dirname(args.args_file)
    else:
        model_dir = os.path.dirname(args.checkpoint)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    model, run_args = load_simple_unet(model_dir, args.checkpoint, device)
    if args.num_inference_steps is not None:
        run_args.num_inference_steps = args.num_inference_steps

    detector, detector_threshold = load_detector(args.detector_checkpoint, device)
    threshold = detector_threshold if args.threshold is None else args.threshold
    trigger_class = getattr(run_args, "trigger_class", run_args.num_classes)
    normal_class_ids = build_normal_class_ids(
        num_classes=run_args.num_classes,
        trigger_class=trigger_class,
        num_samples=args.num_samples,
    )

    normal_images = generate_conditioned_samples(
        model,
        run_args,
        class_labels=normal_class_ids,
        num_samples=args.num_samples,
        device=device,
    )
    trigger_images = generate_conditioned_samples(
        model,
        run_args,
        class_labels=[trigger_class] * args.num_samples,
        num_samples=args.num_samples,
        device=device,
    )

    normal_scores = score_images(detector, normal_images, device)
    trigger_scores = score_images(detector, trigger_images, device)

    if args.save_samples_dir:
        save_samples(normal_images, args.save_samples_dir, "normal")
        save_samples(trigger_images, args.save_samples_dir, "trigger")

    report = build_logo_eval_report(
        checkpoint=args.checkpoint,
        detector_checkpoint=args.detector_checkpoint,
        trigger_scores=trigger_scores,
        normal_scores=normal_scores,
        threshold=threshold,
        num_inference_steps=getattr(run_args, "num_inference_steps", 1000),
        trigger_class=trigger_class,
    )

    output_path = args.output or os.path.join(model_dir, "logo_eval_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Trigger Success Rate: {report['trigger_success_rate']:.4f}")
    print(f"Normal False Positive Rate: {report['normal_false_positive_rate']:.4f}")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
