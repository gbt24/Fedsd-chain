# -*- coding: UTF-8 -*-
import argparse
import os

from torchvision.utils import save_image

from eval_logo_watermark import generate_conditioned_samples, load_simple_unet
from logo_eval_utils import build_normal_class_ids, build_sample_filenames


def main():
    parser = argparse.ArgumentParser(
        description="Generate normal-condition samples from a SimpleUNet checkpoint"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    model_dir = os.path.dirname(args.checkpoint)
    device = "cpu"
    if args.gpu >= 0:
        import torch

        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    model, run_args = load_simple_unet(model_dir, args.checkpoint, device)
    if args.num_inference_steps is not None:
        run_args.num_inference_steps = args.num_inference_steps

    trigger_class = getattr(run_args, "trigger_class", run_args.num_classes)
    normal_class_ids = build_normal_class_ids(
        num_classes=run_args.num_classes,
        trigger_class=trigger_class,
        num_samples=args.num_samples,
    )

    images = generate_conditioned_samples(
        model,
        run_args,
        class_labels=normal_class_ids,
        num_samples=args.num_samples,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for image, file_name in zip(images, build_sample_filenames("normal", len(images))):
        save_image(image, os.path.join(args.output_dir, file_name))

    print(f"Saved {len(images)} normal samples to {args.output_dir}")


if __name__ == "__main__":
    main()
