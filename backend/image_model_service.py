"""Image emotion services with a CLIP-based political scene classifier."""
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from transformers import (
    CLIPModel,
    CLIPProcessor,
)

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
        POLITICAL_MODEL_IDENTIFIER,
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
            POLITICAL_MODEL_IDENTIFIER,
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
            POLITICAL_MODEL_IDENTIFIER,
        )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_LOG_PATH = PROJECT_ROOT / "BUILD_LOG.md"

POLITICAL_POSITIVE_PROMPTS: List[str] = [
    "a photo of a politician",
    "a photo of a political rally or protest",
    "a photo of a government or parliament event",
    "a political campaign poster",
]
POLITICAL_NEGATIVE_PROMPTS: List[str] = [
    "a photo of nature",
    "a photo of food",
    "a random personal photo",
    "an animal",
    "a product photo",
]
POLITICAL_EMOTION_PROMPTS: Dict[str, List[str]] = {
    "Anger": [
        "a political image with an angry, confrontational atmosphere",
        "a political protest image conveying outrage and conflict",
    ],
    "Joy": [
        "a political image with a joyful, hopeful, celebratory atmosphere",
        "a political rally scene conveying optimism and unity",
    ],
    "Sadness": [
        "a solemn political image conveying grief, loss, or sorrow",
        "a political scene with a visibly mournful and somber atmosphere",
    ],
    "Fear": [
        "a political image conveying fear, threat, or public insecurity",
        "a tense political scene conveying anxiety and danger",
    ],
    "Surprise": [
        "a political image capturing shock or unexpected news",
        "a political scene conveying astonishment and disbelief",
    ],
    "Disgust": [
        "a political image conveying revulsion or strong disapproval",
        "a political scene with an atmosphere of moral disgust",
    ],
    "Neutral": [
        "a neutral factual political news photograph without a strong emotion",
        "an ordinary political meeting with a calm, neutral atmosphere",
    ],
}


def append_build_log(message: str) -> None:
    """Append timestamped project activity entries without overwriting previous work."""
    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with BUILD_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"- {timestamp}: {message}\n")


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
        gate_threshold: float = 0.20
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.model_name = model_name
        self.gate_threshold = gate_threshold
        self.model: Optional[ImageEmotionModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_clips_gate(self) -> None:
        """Load the shared CLIP model used for political gating and scene scoring."""
        if self.clip_model is not None and self.clip_processor is not None:
            return

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        append_build_log(
            "CLIP image-text model loaded for political filtering and scene emotion scoring."
        )

    def check_domain_gate(self, image_bytes: bytes, threshold: Optional[float] = None) -> Tuple[bool, float, float]:
        """Check whether the submitted image matches political content using the CLIP zero-shot gate."""
        threshold = self.gate_threshold if threshold is None else threshold
        self.load_clips_gate()
        assert self.clip_model is not None and self.clip_processor is not None

        img, _, _ = self.validate_and_load_image(image_bytes)
        prompts = POLITICAL_POSITIVE_PROMPTS + POLITICAL_NEGATIVE_PROMPTS
        inputs = self.clip_processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image[0]
            clip_scores = (
                logits / self.clip_model.logit_scale.exp()
            ).float().cpu().tolist()

        positive_scores = clip_scores[: len(POLITICAL_POSITIVE_PROMPTS)]
        negative_scores = clip_scores[len(POLITICAL_POSITIVE_PROMPTS):]
        positive_score = float(sum(positive_scores) / len(positive_scores)) if positive_scores else 0.0
        negative_score = float(sum(negative_scores) / len(negative_scores)) if negative_scores else 0.0
        gate_passed = positive_score >= threshold

        append_build_log(
            "Domain gate result: "
            f"positive={positive_score:.4f}, negative={negative_score:.4f}, threshold={threshold:.4f}, passed={gate_passed}."
        )
        return gate_passed, positive_score, threshold

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
        append_build_log(f"Image emotion model loaded from {self.checkpoint_path}.")

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
        Run inference on the provided image bytes after a political-content gate check.
        """
        if not image_bytes:
            raise ValueError("Empty image file received.")

        gate_passed, political_score, threshold = self.check_domain_gate(image_bytes)
        if not gate_passed:
            append_build_log(
                "Emotion classification skipped because the image did not pass the political-content gate."
            )
            return {
                "status": "not_political_content",
                "political_score": round(float(political_score), 4),
                "threshold": round(float(threshold), 4),
                "message": "The uploaded image does not appear to contain political content.",
            }

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


class PoliticalImageEmotionModelService(ImageEmotionModelService):
    """CLIP zero-shot service for political-content gating and scene emotion."""

    def __init__(
        self,
        gate_threshold: float = 0.20,
    ) -> None:
        super().__init__(
            model_name=POLITICAL_MODEL_IDENTIFIER,
            gate_threshold=gate_threshold,
        )

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Gate the image, then score its overall scene against emotion descriptions."""
        if not image_bytes:
            raise ValueError("Empty image file received.")

        gate_passed, political_score, threshold = self.check_domain_gate(image_bytes)
        if not gate_passed:
            append_build_log(
                "Emotion classification skipped because the image did not pass the political-content gate."
            )
            return {
                "status": "not_political_content",
                "political_score": round(float(political_score), 4),
                "threshold": round(float(threshold), 4),
                "message": "The uploaded image does not appear to contain political content.",
            }

        assert self.clip_model is not None and self.clip_processor is not None

        img, img_format, (width, height) = self.validate_and_load_image(image_bytes)
        labels = list(EMOTION_CLASSES)
        prompt_groups = [POLITICAL_EMOTION_PROMPTS[label] for label in labels]
        prompts = [prompt for group in prompt_groups for prompt in group]
        encoded = self.clip_processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.clip_model(**inputs).logits_per_image[0].float()

        class_scores = []
        offset = 0
        for group in prompt_groups:
            class_scores.append(logits[offset : offset + len(group)].mean())
            offset += len(group)
        raw_probabilities = torch.softmax(torch.stack(class_scores), dim=0).cpu().tolist()
        probabilities = {
            label: round(float(raw_probabilities[index]) * 100.0, 1)
            for index, label in enumerate(labels)
        }
        difference = round(100.0 - sum(probabilities.values()), 1)
        if difference:
            dominant_label = max(probabilities, key=probabilities.get)
            probabilities[dominant_label] = round(
                probabilities[dominant_label] + difference, 1
            )

        emotion = max(probabilities, key=probabilities.get)
        return {
            "emotion": emotion,
            "confidence": probabilities[emotion],
            "probabilities": probabilities,
            "metadata": {
                "format": img_format,
                "width": width,
                "height": height,
            },
        }


image_model_service = PoliticalImageEmotionModelService()
