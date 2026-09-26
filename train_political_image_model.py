import argparse
from collections import Counter
from pathlib import Path
import time

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.image_config import EMOTION_CLASSES, PROJECT_ROOT
from backend.image_model_service import append_build_log

DEFAULT_POLITICAL_DATA_DIR = PROJECT_ROOT / "data" / "political_emotion"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "image" / "political_emotion_vit_best.pth"
DEFAULT_EMOTION_LABELS = list(EMOTION_CLASSES)


def available_dataset_labels(data_dir: Path):
    labels = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and item.name in DEFAULT_EMOTION_LABELS:
            labels.append(item.name)
    if not labels:
        labels = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    return labels


class PoliticalEmotionDataset(Dataset):
    def __init__(self, image_files, labels, processor, image_size=224):
        self.image_files = image_files
        self.labels = labels
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        image = __import__("PIL").Image.open(image_path).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"][0]
        return pixel_values, label


def get_label_order(data_dir: Path):
    labels = available_dataset_labels(data_dir)
    if not labels:
        raise FileNotFoundError(f"No emotion label directories found under {data_dir}")

    for label in labels:
        if not (data_dir / label).exists():
            raise FileNotFoundError(f"Dataset folder missing for label {label}")
    return labels


def build_dataset_splits(data_dir: Path, test_size: float = 0.2, seed: int = 42):
    labels = get_label_order(data_dir)
    image_files = []
    label_ids = []

    for label in labels:
        files = sorted((data_dir / label).glob("*.jpg"))
        files.extend(sorted((data_dir / label).glob("*.jpeg")))
        files.extend(sorted((data_dir / label).glob("*.png")))
        if not files:
            continue
        image_files.extend(files)
        label_ids.extend([label] * len(files))

    if not image_files:
        raise FileNotFoundError(f"No images found under {data_dir}")

    train_files, val_files, train_labels, val_labels = train_test_split(
        image_files,
        label_ids,
        test_size=test_size,
        stratify=label_ids,
        random_state=seed,
    )

    return train_files, train_labels, val_files, val_labels, labels


def compute_class_weights(labels, label_order):
    counts = Counter(labels)
    weights = [counts.get(label, 1) for label in label_order]
    max_count = max(weights) if weights else 1
    weight_values = torch.tensor([max_count / max(c, 1) for c in weights], dtype=torch.float32)
    return weight_values


def train_model(
    data_dir: str = str(DEFAULT_POLITICAL_DATA_DIR),
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_path: str = str(DEFAULT_CHECKPOINT_PATH),
    patience: int = 3,
):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Political dataset not found: {data_path}")

    append_build_log(f"Political dataset prep started for {data_path}.")
    train_files, train_labels, val_files, val_labels, label_order = build_dataset_splits(data_path)
    append_build_log(
        "Dataset split created: "
        f"{len(train_files)} train images and {len(val_files)} validation images across {len(label_order)} labels."
    )

    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    train_label_ids = torch.tensor([label_to_index[label] for label in train_labels], dtype=torch.long)
    val_label_ids = torch.tensor([label_to_index[label] for label in val_labels], dtype=torch.long)

    class_weights = compute_class_weights(train_labels, label_order).to("cuda" if torch.cuda.is_available() else "cpu")
    append_build_log(f"Training label order detected: {label_order}.")

    processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
    model = AutoModelForImageClassification.from_pretrained(
        "trpakov/vit-face-expression",
        num_labels=len(label_order),
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for name, param in model.named_parameters():
        if "classifier" in name or "vit.encoder.layer.10" in name or "vit.encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_dataset = PoliticalEmotionDataset(train_files, train_label_ids.tolist(), processor)
    val_dataset = PoliticalEmotionDataset(val_files, val_label_ids.tolist(), processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    save_dest = Path(save_path)
    save_dest.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = train_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - start

        append_build_log(
            f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, elapsed={elapsed:.1f}s."
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "state_dict": model.state_dict(),
                "label_order": label_order,
                "base_model": "trpakov/vit-face-expression",
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            epochs_without_improvement = 0
            torch.save(best_state, save_dest)
            append_build_log(f"Best checkpoint saved to {save_dest} at epoch {epoch}.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                append_build_log(f"Early stopping triggered at epoch {epoch} due to no val_loss improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {save_dest}")
    return save_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a political image emotion classifier with a CLIP domain gate.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_POLITICAL_DATA_DIR), help="Folder containing label-named folders of political images")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path to save the best model checkpoint")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience in validation epochs")
    args = parser.parse_args()

    append_build_log("Political model training script launched.")
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        patience=args.patience,
    )
