from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from config.config import VideoAnalysisResult
from inference.predictor import VideoPredictor
from model.model_interface import VideoModelInterface
from preprocessing.video_reader import VideoReader


class VideoAnalysisService:
    """Service layer for video emotion analysis."""

    def __init__(
        self,
        model: Optional[VideoModelInterface] = None,
        predictor: Optional[VideoPredictor] = None,
    ) -> None:
        self.model = model
        self.predictor = predictor
        self.reader = VideoReader()

        if self.predictor is None and self.model is not None:
            self.predictor = VideoPredictor(
                model=self.model
            )

    def analyze_video(
        self,
        video_path: str | Path,
    ) -> VideoAnalysisResult:
        """Analyze a video and return a structured result."""
        path = Path(video_path)

        if not path.exists():
            return VideoAnalysisResult(
                modality="video",
                status="error",
                message=(
                    f"Video file does not exist: {path}"
                ),
                frames_processed=0,
                duration_seconds=0.0,
                temporal_results=[],
            )

        if self.predictor is None:
            return VideoAnalysisResult(
                modality="video",
                status="model_not_configured",
                message=(
                    "Video emotion model is not configured."
                ),
                frames_processed=0,
                duration_seconds=0.0,
                temporal_results=[],
            )

        try:
            return self.predictor.predict_video(
                path
            )

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as exc:
            return VideoAnalysisResult(
                modality="video",
                status="error",
                message=str(exc),
                frames_processed=0,
                duration_seconds=0.0,
                temporal_results=[],
            )

    def validate_video(
        self,
        video_path: str | Path,
    ) -> dict[str, Any]:
        """Validate a video and return its metadata."""
        path = Path(video_path)

        if not path.exists():
            return {
                "valid": False,
                "error": (
                    f"Video file does not exist: {path}"
                ),
            }

        try:
            metadata = self.reader.read_metadata(
                path
            )

            if not metadata.is_valid:
                return {
                    "valid": False,
                    "error": metadata.error,
                }

            return {
                "valid": True,
                "path": str(path),
                "duration_seconds": float(
                    metadata.duration_seconds
                ),
                "fps": float(
                    metadata.fps
                ),
                "frame_count": int(
                    metadata.frame_count
                ),
                "width": int(
                    metadata.width
                ),
                "height": int(
                    metadata.height
                ),
            }

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
        ) as exc:
            return {
                "valid": False,
                "error": str(exc),
            }

    def get_model_status(
        self,
    ) -> dict[str, Any]:
        """Return the current model status."""
        if self.predictor is None:
            return {
                "configured": False,
                "status": "model_not_configured",
            }

        model = self.predictor.model

        if model is None:
            return {
                "configured": False,
                "status": "model_not_configured",
            }

        if hasattr(
            model,
            "get_model_status",
        ):
            return model.get_model_status()

        return {
            "configured": True,
            "status": "configured",
        }


class VideoAPI(VideoAnalysisService):
    """Backward-compatible alias for VideoAnalysisService."""

    pass


def create_video_analysis_service(
    model: Optional[VideoModelInterface] = None,
    predictor: Optional[VideoPredictor] = None,
) -> VideoAnalysisService:
    """Create a video analysis service."""
    return VideoAnalysisService(
        model=model,
        predictor=predictor,
    )


def create_video_api(
    model: Optional[VideoModelInterface] = None,
    predictor: Optional[VideoPredictor] = None,
) -> VideoAPI:
    """Create a VideoAPI instance."""
    return VideoAPI(
        model=model,
        predictor=predictor,
    )


def analyze_video(
    video_path: str | Path,
    model: Optional[VideoModelInterface] = None,
    predictor: Optional[VideoPredictor] = None,
) -> VideoAnalysisResult:
    """Convenience function for video analysis."""
    service = create_video_analysis_service(
        model=model,
        predictor=predictor,
    )

    return service.analyze_video(
        video_path
    )