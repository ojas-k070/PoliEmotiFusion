import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.image_config import (
    DEFAULT_CHECKPOINT_PATH,
    EMOTION_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)
from backend.image_model_service import ImageEmotionModel

def get_data_loaders(data_dir: Path, batch_size: int = 16, num_workers: int = 0):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Train or val directory not found in {data_dir}")

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)

def train_model(
    data_dir: str = "data/image_emotion",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    save_path: str = str(DEFAULT_CHECKPOINT_PATH),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_path = Path(data_dir)
    train_loader, val_loader, n_train, n_val = get_data_loaders(data_path, batch_size=batch_size, num_workers=0)
    print(f"Dataset loaded: {n_train} train images, {n_val} val images.")

    model = ImageEmotionModel(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    save_dest = Path(save_path)
    save_dest.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()

        scheduler.step()

        train_loss = running_loss / max(n_train, 1)
        train_acc = correct_train / max(n_train, 1)

        model.eval()
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()

        val_loss = val_loss / max(n_val, 1)
        val_acc = correct_val / max(n_val, 1)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.1}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.1}%"
        )

        if val_acc >= best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                save_dest,
            )
            print(f"  => Checkpoint saved to {save_dest}")

    print(f"Training finished! Best Validation Accuracy: {best_val_acc * 100:.1}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B2 Emotion Model")
    parser.add_argument("--data-dir", type=str, default="data/image_emotion", help="Path to image data directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Checkpoint destination")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
