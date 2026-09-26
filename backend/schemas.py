from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        description=(
            "Input political text to analyze. Must have stripped "
            "length between 20 and 1000 characters."
        ),
    )

    category: str = Field(
        default="Other",
        description=(
            "Political category (e.g., Political Speech, "
            "Election Campaign, Other)."
        ),
    )


class TextAnalysisResponse(BaseModel):
    id: str = Field(
        ...,
        description="Unique analysis identifier",
    )

    modality: Literal["text"] = Field(
        default="text",
        description="Input modality, always 'text'",
    )

    emotion: str = Field(
        ...,
        description="Dominant detected emotion label",
    )

    confidence: float = Field(
        ...,
        description=(
            "Confidence percentage for dominant emotion "
            "(0.0 to 100.0)"
        ),
    )

    probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Emotion probability distribution percentages "
            "for all classes, totaling 100"
        ),
    )

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp",
    )

    category: str = Field(
        ...,
        description="Category context associated with the input",
    )

    model: str = Field(
        ...,
        description="Model identifier used for analysis",
    )

    inputLabel: str = Field(
        ...,
        description="60-character preview plus ' — N chars'",
    )

    summary: str = Field(
        ...,
        description="Predominant emotion summary string",
    )

    demo: bool = Field(
        default=False,
        description="Flag indicating if result is demo data (always false)",
    )


class ImageAnalysisResponse(BaseModel):
    id: str = Field(
        ...,
        description="Unique analysis identifier",
    )

    modality: Literal["image"] = Field(
        default="image",
        description="Input modality, always 'image'",
    )

    emotion: str = Field(
        ...,
        description="Dominant detected emotion label",
    )

    confidence: float = Field(
        ...,
        description=(
            "Confidence percentage for dominant emotion "
            "(0.0 to 100.0)"
        ),
    )

    probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Emotion probability distribution percentages "
            "for all classes, totaling 100"
        ),
    )

    timestamp: str = Field(
        ...,
        description="ISO 8601 formatted timestamp",
    )

    category: str = Field(
        ...,
        description="Category context associated with the input",
    )

    model: str = Field(
        ...,
        description="Model identifier used for analysis",
    )

    inputLabel: str = Field(
        ...,
        description="Uploaded image filename and metadata",
    )

    summary: str = Field(
        ...,
        description="Predominant emotion summary string",
    )

    facesDetected: int | None = Field(
        default=None,
        description="Number of detected faces when available",
    )

    demo: bool = Field(
        default=False,
        description="Flag indicating if result is demo data",
    )


class VideoAnalysisResponse(BaseModel):
    modality: Literal["video"] = Field(
        default="video",
        description="Input modality, always 'video'",
    )

    status: str = Field(
        ...,
        description="Status of the video analysis",
    )

    message: str | None = Field(
        default=None,
        description="Optional status or informational message",
    )

    dominant_emotion: str | None = Field(
        default=None,
        description="Dominant emotion detected in the video",
    )

    confidence: float | None = Field(
        default=None,
        description=(
            "Confidence of the dominant emotion, "
            "represented as a value between 0.0 and 1.0"
        ),
    )

    emotions: Dict[str, float] | None = Field(
        default=None,
        description=(
            "Emotion probability distribution with values "
            "between 0.0 and 1.0"
        ),
    )

    frames_processed: int = Field(
        default=0,
        description="Number of video frames processed",
    )

    duration_seconds: float = Field(
        default=0.0,
        description="Duration of the analyzed video in seconds",
    )

    temporal_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Temporal emotion predictions across the video",
    )


class HealthResponse(BaseModel):
    status: str = Field(
        default="ok",
        description="Server health status",
    )