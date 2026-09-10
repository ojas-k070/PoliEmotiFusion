"""
Image Emotion Analysis model service using EfficientNet-B2 in PyTorch.
"""
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

try:
    from backend.image_config import (
        DEFAULT_CHECKPOINT_PATH,
        EMOTION_CLASSES,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        MAX_IMAGE_BYTES,
        MODEL_IDENTIFIER,
        NUM_CLASSES,
    )
except ImportError:
    try:
        from .image_config import (
            DEFAULT_CHECKPOINT_PATH,
            EMOTION_CLASSES,
            IMAGE_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
            MAX_IMAGE_BYTES,
            MODEL_IDENTIFIER,
            NUM_CLASSES,
        )
    except ImportError:
        from image_config import (
            DEFAULT_CHECKPOINT_PATH,
            EMOTION_CLASSES,
            IMAGE_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
            MAX_IMAGE_BYTES,
            MODEL_IDENTIFIER,
            NUM_CLASSES,
        )


class ImageEmotionModel(nn.Module):
    """
    EfficientNet-B2 architecture customized for 7-class emotion classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.efficientnet_b2(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b2(weights=None)


        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ImageEmotionModelService:
    """
    Service to load, preprocess, and run inference on political images.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_name: str = MODEL_IDENTIFIER,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH

        self.model_name = model_name
        self.model: Optional[ImageEmotionModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_model(self) -> None:
        """
        Load fine-tuned model checkpoint if present, or initialize base pretrained model.
        """
        if self.model is not None:
            return

        model = ImageEmotionModel(num_classes=NUM_CLASSES, pretrained=True)

        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict)
            except Exception:
                pass

        model.to(self.device)
        model.eval()
        self.model = model

    def validate_and_load_image(
        self, image_bytes: bytes
    ) -> Tuple[Image.Image, str, Tuple[int, int]]:
        """
        Validate bytes, verify image integrity, and return RGB PIL Image with format and size.
        """
        if not image_bytes:
            raise ValueError("Empty image file received.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            max_mb = MAX_IMAGE_BYTES // (1024 * 1024)
            raise ValueError(f"Image exceeds maximum size limit of {max_mb} MB.")

        try:
            img_io = io.BytesIO(image_bytes)
            with Image.open(img_io) as test_img:
                test_img.verify()
                img_format = test_img.format or "JPEG"

            img_io.seek(0)
            img = Image.open(img_io)
            dimensions = (img.width, img.height)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img, img_format, dimensions
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError("Invalid or corrupted image file. Please provide a valid JPEG, PNG, or WebP image.") from e

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run inference on the provided image bytes.
        """
        if self.model is None:
            self.load_model()

        img, img_format, (width, height) = self.validate_and_load_image(image_bytes)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            raw_probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        raw_percentages: Dict[str, float] = {}
        for idx, prob in enumerate(raw_probs):
            class_name = EMOTION_CLASSES[idx] if idx < len(EMOTION_CLASSES) else str(idx)
            raw_percentages[class_name] = prob * 100.0

        for label in EMOTION_CLASSES:
            if label not in raw_percentages:
                raw_percentages[label] = 0.0

        probabilities: Dict[str, float] = {
            label: round(raw_percentages[label], 1) for label in EMOTION_CLASSES
        }

        current_sum = round(sum(probabilities.values()), 1)
        diff = round(100.0 - current_sum, 1)
        if diff != 0:
            dominant_key = max(probabilities, key=probabilities.get)
            probabilities[dominant_key] = round(probabilities[dominant_key] + diff, 1)

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "metadata": {
                "format": img_format,
                "width": width,
                "height": height,
            },
        }


image_model_service = ImageEmotionModelService()
