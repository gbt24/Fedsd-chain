# -*- coding: UTF-8 -*-
import argparse
import json
import os
import random

import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class BinaryImageFolderDataset(Dataset):
    def __init__(self, image_paths, label, transform):
        self.image_paths = list(image_paths)
        self.label = float(label)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor([self.label], dtype=torch.float32)


class BinaryWrappedDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = float(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, torch.tensor([self.label], dtype=torch.float32)


class LogoDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def list_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file_name))
    image_paths.sort()
    return image_paths


def build_positive_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def sample_dataset(dataset, max_items, seed):
    if max_items is None or len(dataset) <= max_items:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def build_negative_dataset(args, transform):
    negative_datasets = []

    cifar_dataset = BinaryWrappedDataset(
        datasets.CIFAR10(
            "./data/cifar10/",
            train=False,
            download=True,
            transform=transform,
        ),
        label=0,
    )
    negative_datasets.append(sample_dataset(cifar_dataset, args.max_real_negatives, args.seed))

    if args.generated_normal_dir:
        generated_paths = list_image_paths(args.generated_normal_dir)
        if generated_paths:
            negative_datasets.append(
                BinaryImageFolderDataset(generated_paths, label=0, transform=transform)
            )

    if len(negative_datasets) == 1:
        return negative_datasets[0]
    return ConcatDataset(negative_datasets)


def split_dataset(dataset, val_ratio, seed):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    true_positive = 0
    true_negative = 0
    positive_total = 0
    negative_total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            correct += (preds == labels).sum().item()

            positive_mask = labels == 1
            negative_mask = labels == 0
            true_positive += ((preds == 1) & positive_mask).sum().item()
            true_negative += ((preds == 0) & negative_mask).sum().item()
            positive_total += positive_mask.sum().item()
            negative_total += negative_mask.sum().item()

    positive_recall = true_positive / max(positive_total, 1)
    negative_recall = true_negative / max(negative_total, 1)
    balanced_accuracy = (positive_recall + negative_recall) / 2.0
    return total_loss / max(total, 1), correct / max(total, 1), balanced_accuracy


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    positive_paths = list_image_paths(args.logo_dir)
    if not positive_paths:
        raise FileNotFoundError(f"No logo images found in {args.logo_dir}")

    positive_transform = build_positive_train_transform(args.image_size)
    eval_transform = build_eval_transform(args.image_size)
    positive_dataset = BinaryImageFolderDataset(
        positive_paths, label=1, transform=positive_transform
    )
    negative_dataset = build_negative_dataset(args, eval_transform)
    max_negative_samples = args.max_negative_samples
    if max_negative_samples is None:
        max_negative_samples = max(len(positive_dataset) * args.negative_ratio, 1)
    negative_dataset = sample_dataset(negative_dataset, max_negative_samples, args.seed + 1)

    dataset = ConcatDataset([positive_dataset, negative_dataset])
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    model = LogoDetector().to(device)
    positive_weight = torch.tensor(
        [max(len(negative_dataset), 1) / max(len(positive_dataset), 1)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_balanced_acc = -1.0
    best_model_path = os.path.join(args.output_dir, "model_best.pth")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = total_loss / max(total, 1)
        val_loss, val_acc, val_balanced_acc = evaluate(
            model, val_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_balanced_acc,
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_balanced_acc:.4f}"
        )

        if val_balanced_acc >= best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "threshold": 0.5,
                },
                best_model_path,
            )

    summary = {
        "logo_dir": args.logo_dir,
        "generated_normal_dir": args.generated_normal_dir,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_val_balanced_accuracy": best_val_balanced_acc,
        "num_positive_samples": len(positive_dataset),
        "num_negative_samples": len(negative_dataset),
        "history": history,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved best detector to {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a binary logo detector")
    parser.add_argument("--logo_dir", type=str, required=True)
    parser.add_argument("--generated_normal_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./result/logo_detector/")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_real_negatives", type=int, default=5000)
    parser.add_argument("--max_negative_samples", type=int, default=None)
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
