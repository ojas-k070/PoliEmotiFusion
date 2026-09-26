"""
Configuration settings and constants for the Image Emotion Analysis module.
"""
from pathlib import Path
from typing import List

# Target 7 emotion classes matching the project-wide taxonomy
EMOTION_CLASSES: List[str] = [
    "Anger",
    "Joy",
    "Sadness",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]

NUM_CLASSES: int = len(EMOTION_CLASSES)

# Label index mappings
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(EMOTION_CLASSES)}

# Normalization and input specifications for EfficientNet-B2
IMAGE_SIZE: int = 260
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model naming and paths
MODEL_IDENTIFIER: str = "EfficientNet-B2 Emotion Classifier"
POLITICAL_MODEL_IDENTIFIER: str = "CLIP Zero-Shot Political Scene Emotion"
POLITICAL_BASE_MODEL: str = "trpakov/vit-face-expression"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "image" / "efficientnet_b2_emotion.pth"
POLITICAL_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "image" / "political_emotion_vit_best.pth"
DATA_DIR = PROJECT_ROOT / "data" / "image_emotion"

# File upload restrictions
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
]
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
