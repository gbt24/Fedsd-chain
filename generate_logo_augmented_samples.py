# -*- coding: UTF-8 -*-
import argparse
import os
import random

from PIL import Image, ImageEnhance

from logo_eval_utils import build_augmented_sample_filenames


VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_logo_files(input_dir):
    files = [
        name
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]
    if not files:
        raise FileNotFoundError(f"No logo images found in {input_dir}")
    return files


def apply_random_logo_augmentation(image, image_size, rng):
    image = image.convert("RGBA").resize((image_size, image_size), Image.Resampling.BILINEAR)

    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.88, 1.12)
    translate_x = rng.randint(-2, 2)
    translate_y = rng.randint(-2, 2)

    scaled_size = max(1, int(image_size * scale))
    scaled = image.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    paste_x = (image_size - scaled_size) // 2 + translate_x
    paste_y = (image_size - scaled_size) // 2 + translate_y
    canvas.alpha_composite(scaled, (paste_x, paste_y))
    canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.1:
        canvas = canvas.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    brightness = ImageEnhance.Brightness(canvas)
    canvas = brightness.enhance(rng.uniform(0.9, 1.1))
    contrast = ImageEnhance.Contrast(canvas)
    canvas = contrast.enhance(rng.uniform(0.9, 1.1))
    color = ImageEnhance.Color(canvas)
    canvas = color.enhance(rng.uniform(0.9, 1.1))

    return canvas.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detector-only augmented logo samples from source triggers"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variants_per_image", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    source_files = list_logo_files(args.input_dir)
    output_names = build_augmented_sample_filenames(
        source_files, args.variants_per_image
    )

    output_index = 0
    for file_name in source_files:
        input_path = os.path.join(args.input_dir, file_name)
        source_image = Image.open(input_path)
        for _ in range(args.variants_per_image):
            augmented = apply_random_logo_augmentation(
                source_image, args.image_size, rng
            )
            augmented.save(os.path.join(args.output_dir, output_names[output_index]))
            output_index += 1

    print(
        f"Saved {len(output_names)} augmented logo samples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
