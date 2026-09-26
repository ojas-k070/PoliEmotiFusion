from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv


# Load environment variables from the .env file if available.
load_dotenv()


# Supported emotion labels for the video emotion analysis module.
EMOTION_LABELS: Tuple[str, ...] = (
    "anger",
    "joy",
    "sadness",
    "fear",
    "surprise",
    "disgust",
    "neutral",
)


# Current pretrained facial-expression model used by the video module.
DEFAULT_VIDEO_MODEL_NAME = "deanngkl/vit-tiny-fer"


# Default resolution used during video preprocessing.
DEFAULT_TARGET_RESOLUTION: Tuple[int, int] = (224, 224)


# Supported video file extensions.
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".flv",
}


@dataclass
class VideoAnalysisResult:
    """Standardized output schema for video analysis."""

    modality: str = "video"

    status: str = "model_not_configured"

    message: str | None = (
        "Video emotion model is not configured yet."
    )

    dominant_emotion: str | None = None

    confidence: float | None = None

    emotions: Dict[str, float] | None = None

    frames_processed: int = 0

    duration_seconds: float = 0.0

    temporal_results: list[Dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result into a dictionary."""

        return {
            "modality": self.modality,
            "status": self.status,
            "message": self.message,
            "dominant_emotion": self.dominant_emotion,
            "confidence": (
                float(self.confidence)
                if self.confidence is not None
                else None
            ),
            "emotions": (
                {
                    key: float(value)
                    for key, value in self.emotions.items()
                }
                if self.emotions is not None
                else None
            ),
            "frames_processed": int(self.frames_processed),
            "duration_seconds": float(self.duration_seconds),
            "temporal_results": self.temporal_results,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "VideoAnalysisResult":
        """Create a VideoAnalysisResult from a dictionary."""

        emotions = data.get("emotions")

        if emotions is not None:
            if not isinstance(emotions, dict):
                raise ValueError(
                    "'emotions' must be a dictionary."
                )

            emotion_map: Dict[str, float] = {}

            for label in EMOTION_LABELS:
                if label in emotions:
                    value = float(emotions[label])

                    if value < 0.0 or value > 1.0:
                        raise ValueError(
                            f"Emotion probability for '{label}' "
                            f"must be between 0.0 and 1.0."
                        )

                    emotion_map[label] = value

            emotions = emotion_map

        confidence = data.get("confidence")

        if confidence is not None:
            confidence = float(confidence)

            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(
                    "'confidence' must be between 0.0 and 1.0."
                )

        frames_processed = int(
            data.get("frames_processed", 0)
        )

        if frames_processed < 0:
            raise ValueError(
                "'frames_processed' cannot be negative."
            )

        duration_seconds = float(
            data.get("duration_seconds", 0.0)
        )

        if duration_seconds < 0.0:
            raise ValueError(
                "'duration_seconds' cannot be negative."
            )

        temporal_results = data.get(
            "temporal_results",
            [],
        )

        if not isinstance(temporal_results, list):
            raise ValueError(
                "'temporal_results' must be a list."
            )

        return cls(
            modality=data.get(
                "modality",
                "video",
            ),
            status=data.get(
                "status",
                "model_not_configured",
            ),
            message=data.get("message"),
            dominant_emotion=data.get(
                "dominant_emotion"
            ),
            confidence=confidence,
            emotions=emotions,
            frames_processed=frames_processed,
            duration_seconds=duration_seconds,
            temporal_results=temporal_results,
        )


@dataclass
class VideoConfig:
    """Configuration for video preprocessing and analysis."""

    target_resolution: Tuple[int, int] = (
        DEFAULT_TARGET_RESOLUTION
    )

    # Number of frames to skip between sampled frames.
    sample_rate: int = 1

    # Optional maximum duration to process.
    max_duration_seconds: float | None = None

    # Starting timestamp for video processing.
    start_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate video preprocessing configuration."""

        width, height = self.target_resolution

        if width <= 0 or height <= 0:
            raise ValueError(
                "target_resolution values must be greater than zero."
            )

        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if self.max_duration_seconds is not None:
            if self.max_duration_seconds <= 0:
                raise ValueError(
                    "max_duration_seconds must be greater than zero."
                )

        if self.start_time_seconds < 0:
            raise ValueError(
                "start_time_seconds cannot be negative."
            )


@dataclass
class ModelConfig:
    """Configuration for the current video emotion model."""

    model_name: str = DEFAULT_VIDEO_MODEL_NAME

    num_labels: int = len(EMOTION_LABELS)

    device: str = "auto"


def load_module_config() -> Dict[str, Any]:
    """Load configuration values for the video module.

    Environment variables can be provided through the .env file.
    Existing configuration keys are preserved for compatibility
    with the rest of the video module.
    """

    load_dotenv()

    max_duration_value = os.getenv(
        "VIDEO_MAX_DURATION_SECONDS"
    )

    configured_model_name = os.getenv(
        "VIDEO_MODEL_NAME",
        DEFAULT_VIDEO_MODEL_NAME,
    )

    configured_device = os.getenv(
        "VIDEO_MODEL_DEVICE",
        "auto",
    )

    return {
        # Existing project configuration.
        "youtube_api_key": os.getenv(
            "YOUTUBE_API_KEY",
            "",
        ),

        "video_model_name": configured_model_name,

        "video_model_path": os.getenv(
            "VIDEO_MODEL_PATH",
            "",
        ),

        "video_input_dir": os.getenv(
            "VIDEO_INPUT_DIR",
            str(
                Path("./data/raw").resolve()
            ),
        ),

        "video_output_dir": os.getenv(
            "VIDEO_OUTPUT_DIR",
            str(
                Path("./data/processed").resolve()
            ),
        ),

        # Emotion configuration.
        "emotion_labels": list(EMOTION_LABELS),

        # Supported video formats.
        "supported_video_extensions": sorted(
            SUPPORTED_VIDEO_EXTENSIONS
        ),

        # Video preprocessing configuration.
        "default_target_resolution": list(
            DEFAULT_TARGET_RESOLUTION
        ),

        "target_resolution": DEFAULT_TARGET_RESOLUTION,

        "sample_rate": int(
            os.getenv(
                "VIDEO_SAMPLE_RATE",
                "1",
            )
        ),

        "max_duration_seconds": (
            float(max_duration_value)
            if max_duration_value
            else None
        ),

        "start_time_seconds": float(
            os.getenv(
                "VIDEO_START_TIME_SECONDS",
                "0",
            )
        ),

        # Current pretrained model configuration.
        "model_name": configured_model_name,

        "device": configured_device,
    }


def get_module_settings() -> Dict[str, Any]:
    """Return the complete video module settings."""

    return load_module_config()


def get_env_config() -> Dict[str, Any]:
    """Return model-related environment configuration."""

    return {
        "model_name": os.getenv(
            "VIDEO_MODEL_NAME",
            DEFAULT_VIDEO_MODEL_NAME,
        ),

        "device": os.getenv(
            "VIDEO_MODEL_DEVICE",
            "auto",
        ),
    }