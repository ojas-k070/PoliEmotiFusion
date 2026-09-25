from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np
import torch

from config.config import EMOTION_LABELS
from .model_interface import VideoModelInterface


class VideoMAEAdapter(VideoModelInterface):
    """Lightweight pretrained facial-emotion model for video analysis.

    The existing class name is retained for compatibility with the project's
    current API. Internally, this adapter uses the pretrained ViT-Tiny
    TorchScript facial-expression model instead of an untrained VideoMAE
    classification head.

    Each sampled video frame is classified independently and the resulting
    emotion probabilities are aggregated into a video-level prediction.

    This approach does not require a project-specific labeled video dataset
    or fine-tuning before inference.
    """

    DEFAULT_MODEL_NAME = "deanngkl/vit-tiny-fer"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: str | None = None,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = self._resolve_device(device)

        self.model: torch.jit.ScriptModule | None = None

        self.is_loaded = False
        self.is_fine_tuned = True

        self.model_labels: List[str] = []

    def _resolve_device(
        self,
        device: str,
    ) -> str:
        """Resolve the requested device."""
        normalized = device.lower().strip()

        if normalized == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if (
            normalized == "cuda"
            and not torch.cuda.is_available()
        ):
            return "cpu"

        if normalized not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device must be 'auto', 'cpu', or 'cuda'."
            )

        return normalized

    def _get_default_model_path(self) -> Path:
        """Find the downloaded TorchScript model in the Hugging Face cache."""
        cache_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--deanngkl--vit-tiny-fer"
            / "snapshots"
        )

        if not cache_root.exists():
            raise FileNotFoundError(
                "The pretrained ViT-Tiny model has not "
                "been downloaded. Run the Hugging Face "
                "download command first."
            )

        candidates = list(
            cache_root.glob(
                "*/emotion_vit_tiny.torchscript.pt"
            )
        )

        if not candidates:
            raise FileNotFoundError(
                "Could not find "
                "emotion_vit_tiny.torchscript.pt "
                "inside the Hugging Face cache."
            )

        return candidates[0]

    def load_model(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Load the pretrained TorchScript emotion model."""
        if self.model_path:
            model_path = Path(
                self.model_path
            )
        else:
            model_path = (
                self._get_default_model_path()
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Emotion model file does not exist: "
                f"{model_path}"
            )

        self.model = torch.jit.load(
            str(model_path),
            map_location=self.device,
        )

        self.model.eval()

        self.model_labels = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]

        self.is_loaded = True
        self.is_fine_tuned = True

    def _prepare_frame(
        self,
        frame: Any,
    ) -> np.ndarray:
        """Convert an input frame into RGB uint8 format."""
        array = np.asarray(frame)

        if array.size == 0:
            raise ValueError(
                "Cannot process an empty video frame."
            )

        if array.ndim != 3:
            raise ValueError(
                "Frame must have shape (H, W, C). "
                f"Received {array.shape}."
            )

        channels = array.shape[2]

        if channels == 1:
            array = np.repeat(
                array,
                3,
                axis=2,
            )

        elif channels == 4:
            array = array[:, :, :3]

        elif channels != 3:
            raise ValueError(
                f"Frame has {channels} channels. "
                "Expected 1, 3, or 4 channels."
            )

        array = array.astype(
            np.float32,
            copy=False,
        )

        if array.max() <= 1.0:
            array *= 255.0

        array = np.clip(
            array,
            0.0,
            255.0,
        )

        # OpenCV frames are BGR.
        # Convert to RGB.
        array = array[:, :, ::-1].copy()

        return array.astype(
            np.uint8,
            copy=False,
        )

    def _prepare_tensor(
        self,
        frame: Any,
    ) -> torch.Tensor:
        """Convert a frame into a normalized model tensor."""
        rgb_frame = self._prepare_frame(
            frame
        )

        tensor = torch.from_numpy(
            rgb_frame
        ).float()

        # HWC -> CHW
        tensor = tensor.permute(
            2,
            0,
            1,
        )

        # Convert pixel values from [0, 255]
        # to [0, 1].
        tensor = tensor / 255.0

        # ImageNet normalization.
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=tensor.dtype,
        ).view(
            3,
            1,
            1,
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=tensor.dtype,
        ).view(
            3,
            1,
            1,
        )

        tensor = (
            tensor - mean
        ) / std

        # Add batch dimension.
        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device
        )

    def _predict_frame(
        self,
        frame: Any,
    ) -> dict[str, float]:
        """Predict emotion probabilities for one frame."""
        if self.model is None:
            raise RuntimeError(
                "Emotion model is not loaded."
            )

        tensor = self._prepare_tensor(
            frame
        )

        with torch.no_grad():
            outputs = self.model(
                tensor
            )

        if isinstance(
            outputs,
            (tuple, list),
        ):
            logits = outputs[0]
        else:
            logits = outputs

        if not isinstance(
            logits,
            torch.Tensor,
        ):
            logits = torch.as_tensor(
                logits,
                device=self.device,
            )

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        if (
            probabilities.shape[-1]
            != len(self.model_labels)
        ):
            raise RuntimeError(
                "Unexpected number of model outputs. "
                f"Expected {len(self.model_labels)}, "
                f"received "
                f"{probabilities.shape[-1]}."
            )

        return {
            label: float(
                probabilities[index].item()
            )
            for index, label in enumerate(
                self.model_labels
            )
        }

    def _aggregate_predictions(
        self,
        predictions: Sequence[
            dict[str, float]
        ],
    ) -> dict[str, Any]:
        """Aggregate frame-level predictions."""
        if len(predictions) == 0:
            raise ValueError(
                "At least one frame prediction "
                "is required."
            )

        totals = {
            label: 0.0
            for label in EMOTION_LABELS
        }

        for prediction in predictions:
            for label in EMOTION_LABELS:
                totals[label] += float(
                    prediction.get(
                        label,
                        0.0,
                    )
                )

        count = len(predictions)

        emotions = {
            label: totals[label] / count
            for label in EMOTION_LABELS
        }

        total = sum(
            emotions.values()
        )

        if total <= 0.0:
            raise RuntimeError(
                "Emotion model produced zero "
                "probability mass."
            )

        emotions = {
            label: value / total
            for label, value in emotions.items()
        }

        dominant_emotion = max(
            emotions,
            key=emotions.get,
        )

        confidence = float(
            emotions[
                dominant_emotion
            ]
        )

        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
        }

    def predict(
        self,
        frames: Sequence[Any],
    ) -> Any:
        """Predict emotion from a sequence of video frames."""
        if (
            not self.is_loaded
            or self.model is None
        ):
            raise RuntimeError(
                "Emotion model is not loaded. "
                "Call load_model() before "
                "calling predict()."
            )

        if frames is None:
            raise ValueError(
                "At least one frame is required."
            )

        if len(frames) == 0:
            raise ValueError(
                "At least one frame is required."
            )

        frame_predictions = [
            self._predict_frame(frame)
            for frame in frames
        ]

        result = (
            self._aggregate_predictions(
                frame_predictions
            )
        )

        return {
            "status": "success",
            "message": (
                "Emotion estimated using a "
                "pretrained facial-expression "
                "recognition model. The result "
                "represents visible facial expression "
                "and may not fully represent the "
                "underlying emotion."
            ),
            "emotions": result[
                "emotions"
            ],
            "dominant_emotion": result[
                "dominant_emotion"
            ],
            "confidence": result[
                "confidence"
            ],
            "temporal_results": [],
        }

    def predict_batch(
        self,
        frames_batch: Iterable[
            Sequence[Any]
        ],
    ) -> List[Any]:
        """Predict emotions for multiple video clips."""
        if (
            not self.is_loaded
            or self.model is None
        ):
            raise RuntimeError(
                "Emotion model is not loaded. "
                "Call load_model() before "
                "calling predict_batch()."
            )

        batch = list(
            frames_batch
        )

        if len(batch) == 0:
            raise ValueError(
                "At least one video frame sequence "
                "is required."
            )

        return [
            self.predict(frames)
            for frames in batch
        ]

    def get_labels(
        self,
    ) -> List[str]:
        """Return the project's seven emotion labels."""
        return list(
            EMOTION_LABELS
        )

    def get_model_status(
        self,
    ) -> dict[str, Any]:
        """Return the current model state."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "is_fine_tuned": self.is_fine_tuned,
            "status": (
                "loaded_pretrained_emotion_model"
                if self.is_loaded
                else "model_not_loaded"
            ),
        }