import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
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


def evaluate(data_dir: str = "data/image_emotion", checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(data_dir) / "test"

    if not test_dir.exists():
        test_dir = Path(data_dir) / "val"
        print(f"Test set not found, falling back to validation set: {test_dir}")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ImageEmotionModel(num_classes=NUM_CLASSES, pretrained=False)
    cp_path = Path(checkpoint_path)

    if cp_path.exists():
        checkpoint = torch.load(cp_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {cp_path}")
    else:
        print("No checkpoint found, evaluating with base initialized model.")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = (all_preds == all_targets).mean() * 100.0
    print(f"\nTest Accuracy: {acc:.2f}%")

    target_names = [EMOTION_CLASSES[i] for i in sorted(set(all_targets).union(set(all_preds)))]
    report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    doc_dir = Path("docs")
    doc_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title="Image Emotion Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    cm_path = doc_dir / "image_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/image_emotion")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()
    evaluate(data_dir=args.data_dir, checkpoint_path=args.checkpoint)
