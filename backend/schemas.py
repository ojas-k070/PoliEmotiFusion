from typing import Dict, List, Literal, Any
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


class SpeechIntelligenceResponse(BaseModel):
    id: str = Field(..., description="Unique speech analysis identifier")
    modality: Literal["audio", "speech"] = Field(default="audio", description="Input modality")
    status: str = Field(default="completed", description="Analysis status")
    language: str = Field(..., description="Detected language code (e.g., 'hi', 'en', 'mr')")
    languageName: str = Field(..., description="Detected language name (e.g., 'Hindi', 'English', 'Marathi')")
    languageConfidence: float = Field(default=1.0, description="Confidence of language detection (0.0 to 1.0)")
    transcript: str = Field(..., description="Original speech transcript in detected language")
    translation: str = Field(..., description="Fluent English translation")
    isTranslationNeeded: bool = Field(default=True, description="Whether translation was needed")
    duration: float = Field(..., description="Audio duration in seconds")
    formattedDuration: str = Field(default="0:00", description="Formatted duration mm:ss")
    chunkCount: int = Field(default=1, description="Number of audio chunks processed")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Per-chunk transcription details")
    audioStats: Dict[str, Any] = Field(default_factory=dict, description="Acoustic file metadata")
    waveform: List[float] = Field(default_factory=list, description="Normalized waveform peaks for player")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    category: str = Field(default="Other", description="Political category context")
    inputLabel: str = Field(..., description="Filename and duration/size label")
    model: str = Field(default="openai/whisper-small", description="Whisper model used")
    processingTime: float = Field(default=0.0, description="Processing time in seconds")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Server health status")
    models: Dict[str, bool] = Field(
        default_factory=dict,
        description="Loaded state of machine learning models",
    )
