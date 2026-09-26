from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        description="Input political text to analyze. Must have stripped length between 20 and 1000 characters.",
    )
    category: str = Field(
        default="Other",
        description="Political category (e.g., Political Speech, Election Campaign, Other).",
    )


class TextAnalysisResponse(BaseModel):
    id: str = Field(..., description="Unique analysis identifier")
    modality: Literal["text"] = Field(default="text", description="Input modality, always 'text'")
    emotion: str = Field(..., description="Dominant detected emotion label")
    confidence: float = Field(..., description="Confidence percentage for dominant emotion (0.0 to 100.0)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Emotion probability distribution percentages for all classes, totaling 100",
    )
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    category: str = Field(..., description="Category context associated with the input")
    model: str = Field(..., description="Model identifier used for analysis")
    inputLabel: str = Field(..., description="60-character preview plus ' — N chars'")
    summary: str = Field(..., description="Predominant emotion summary string")
    demo: bool = Field(default=False, description="Flag indicating if result is demo data (always false)")


class ImageAnalysisResponse(BaseModel):
    id: str = Field(..., description="Unique analysis identifier")
    modality: Literal["image"] = Field(default="image", description="Input modality, always 'image'")
    emotion: str = Field(..., description="Dominant detected emotion label")
    confidence: float = Field(..., description="Normalized model score for the top emotion label (0.0 to 100.0)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Normalized CLIP similarity score percentages across all emotion labels, totaling 100; not calibrated probabilities",
    )
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    category: str = Field(..., description="Category context associated with the input")
    model: str = Field(..., description="Model identifier used for analysis")
    inputLabel: str = Field(..., description="Image filename with dimensions")
    summary: str = Field(..., description="Predominant emotion summary string")
    facesDetected: Optional[int] = Field(default=None, description="Number of detected faces if face detection is enabled")
    demo: bool = Field(default=False, description="Flag indicating if result is demo data (always false)")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server health status")
