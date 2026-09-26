from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..analysis.emotion_aggregator import EmotionAggregator
from ..config.config import VideoAnalysisResult
from ..model.model_interface import VideoModelInterface
from ..preprocessing.preprocess import preprocess_video_frames
from ..preprocessing.video_reader import VideoReader


@dataclass
class VideoPredictor:
    """Run video preprocessing, temporal analysis, and emotion inference."""

    model: Optional[VideoModelInterface] = None
    sample_rate: int = 1
    max_frames: int = 32
    max_duration_seconds: float | None = None
    target_resolution: tuple[int, int] = (224, 224)
    start_time_seconds: float = 0.0
    normalize: bool = True
    temporal_window_size: int = 4

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if self.max_frames <= 0:
            raise ValueError(
                "max_frames must be greater than zero."
            )

        if self.temporal_window_size <= 0:
            raise ValueError(
                "temporal_window_size must be greater than zero."
            )

        if self.start_time_seconds < 0:
            raise ValueError(
                "start_time_seconds cannot be negative."
            )

        if (
            self.max_duration_seconds is not None
            and self.max_duration_seconds <= 0
        ):
            raise ValueError(
                "max_duration_seconds must be greater than zero."
            )

    def predict_video(
        self,
        video_path: str | Path,
    ) -> VideoAnalysisResult:
        """Run the complete video emotion pipeline."""

        if self.model is None:
            return VideoAnalysisResult(
                modality="video",
                status="model_not_configured",
                message=(
                    "Video emotion model is not "
                    "configured yet."
                ),
                dominant_emotion=None,
                confidence=None,
                emotions=None,
                frames_processed=0,
                duration_seconds=0.0,
                temporal_results=[],
            )

        path = Path(video_path)

        reader = VideoReader()
        metadata = reader.read_metadata(path)

        if not metadata.is_valid:
            raise ValueError(
                metadata.error
                or "Video could not be processed."
            )

        frames = preprocess_video_frames(
            path,
            sample_rate=self.sample_rate,
            max_frames=self.max_frames,
            max_duration_seconds=self.max_duration_seconds,
            target_resolution=self.target_resolution,
            start_time_seconds=self.start_time_seconds,
            normalize=self.normalize,
        )

        effective_duration = self._get_effective_duration(
            metadata.duration_seconds
        )

        temporal_results = (
            self._predict_temporal_windows(
                frames=frames,
                duration_seconds=effective_duration,
            )
        )

        overall_prediction = (
            self._aggregate_temporal_results(
                temporal_results
            )
        )

        return VideoAnalysisResult(
            modality="video",
            status=overall_prediction.get(
                "status",
                "success",
            ),
            message=overall_prediction.get(
                "message"
            ),
            dominant_emotion=overall_prediction.get(
                "dominant_emotion"
            ),
            confidence=overall_prediction.get(
                "confidence"
            ),
            emotions=overall_prediction.get(
                "emotions"
            ),
            frames_processed=len(frames),
            duration_seconds=effective_duration,
            temporal_results=temporal_results,
        )

    def predict_frames(
        self,
        frames: Sequence[np.ndarray],
    ) -> VideoAnalysisResult:
        """Predict emotion from an already-loaded frame sequence."""

        if self.model is None:
            return VideoAnalysisResult(
                modality="video",
                status="model_not_configured",
                message=(
                    "Video emotion model is not "
                    "configured yet."
                ),
                dominant_emotion=None,
                confidence=None,
                emotions=None,
                frames_processed=0,
                duration_seconds=0.0,
                temporal_results=[],
            )

        if frames is None:
            raise ValueError(
                "At least one frame is required for prediction."
            )

        if len(frames) == 0:
            raise ValueError(
                "At least one frame is required for prediction."
            )

        temporal_results = (
            self._predict_temporal_windows(
                frames=frames,
                duration_seconds=0.0,
            )
        )

        overall_prediction = (
            self._aggregate_temporal_results(
                temporal_results
            )
        )

        return VideoAnalysisResult(
            modality="video",
            status=overall_prediction.get(
                "status",
                "success",
            ),
            message=overall_prediction.get(
                "message"
            ),
            dominant_emotion=overall_prediction.get(
                "dominant_emotion"
            ),
            confidence=overall_prediction.get(
                "confidence"
            ),
            emotions=overall_prediction.get(
                "emotions"
            ),
            frames_processed=len(frames),
            duration_seconds=0.0,
            temporal_results=temporal_results,
        )

    def _get_effective_duration(
        self,
        video_duration: float,
    ) -> float:
        """Calculate the duration actually analyzed."""

        available_duration = (
            video_duration
            - self.start_time_seconds
        )

        if available_duration <= 0:
            raise ValueError(
                "start_time_seconds is beyond "
                "the end of the video."
            )

        if self.max_duration_seconds is not None:
            available_duration = min(
                available_duration,
                self.max_duration_seconds,
            )

        return float(available_duration)

    def _split_into_windows(
        self,
        frames: Sequence[np.ndarray],
    ) -> list[Sequence[np.ndarray]]:
        """Split sampled frames into ordered temporal windows."""

        if frames is None:
            raise ValueError(
                "Frames cannot be None."
            )

        if len(frames) == 0:
            raise ValueError(
                "At least one frame is required."
            )

        windows: list[Sequence[np.ndarray]] = []

        for start in range(
            0,
            len(frames),
            self.temporal_window_size,
        ):
            end = min(
                start + self.temporal_window_size,
                len(frames),
            )

            window = frames[start:end]

            if len(window) > 0:
                windows.append(window)

        return windows

    def _get_window_time_bounds(
        self,
        window_index: int,
        total_windows: int,
        duration_seconds: float,
    ) -> tuple[float, float, float]:
        """Return start, end, and midpoint for a temporal window."""

        if duration_seconds <= 0:
            timestamp = float(window_index)

            return (
                timestamp,
                timestamp,
                timestamp,
            )

        if total_windows <= 0:
            return (
                0.0,
                duration_seconds,
                duration_seconds / 2.0,
            )

        window_duration = (
            duration_seconds
            / total_windows
        )

        start = (
            window_index
            * window_duration
        )

        end = min(
            start + window_duration,
            duration_seconds,
        )

        timestamp = (
            start
            + (end - start) / 2.0
        )

        return (
            float(start),
            float(end),
            float(timestamp),
        )

    def _predict_temporal_windows(
        self,
        frames: Sequence[np.ndarray],
        duration_seconds: float,
    ) -> list[Dict[str, Any]]:
        """Predict emotion for each temporal window."""

        if self.model is None:
            raise RuntimeError(
                "Video emotion model is not configured."
            )

        windows = self._split_into_windows(
            frames
        )

        temporal_results: list[
            Dict[str, Any]
        ] = []

        for window_index, window in enumerate(
            windows
        ):
            prediction = self.model.predict(
                window
            )

            if not isinstance(
                prediction,
                dict,
            ):
                raise ValueError(
                    "Model prediction must be a dictionary."
                )

            emotions = prediction.get(
                "emotions"
            )

            if emotions is not None:
                if not isinstance(
                    emotions,
                    dict,
                ):
                    raise ValueError(
                        "Prediction 'emotions' "
                        "must be a dictionary."
                    )

                emotions = {
                    str(label): float(value)
                    for label, value in emotions.items()
                }

            dominant_emotion = prediction.get(
                "dominant_emotion"
            )

            confidence = prediction.get(
                "confidence"
            )

            if confidence is not None:
                confidence = float(
                    confidence
                )

            (
                start_seconds,
                end_seconds,
                timestamp,
            ) = self._get_window_time_bounds(
                window_index=window_index,
                total_windows=len(windows),
                duration_seconds=duration_seconds,
            )

            temporal_results.append(
                {
                    "timestamp": timestamp,
                    "timestamp_seconds": timestamp,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "emotions": emotions,
                    "frame_start": (
                        window_index
                        * self.temporal_window_size
                    ),
                    "frame_end": (
                        window_index
                        * self.temporal_window_size
                        + len(window)
                        - 1
                    ),
                    "frames_in_window": len(
                        window
                    ),
                }
            )

        return temporal_results

    def _aggregate_temporal_results(
        self,
        temporal_results: Sequence[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """Aggregate temporal predictions into one video result."""

        if not temporal_results:
            raise ValueError(
                "No temporal predictions were generated."
            )

        emotion_distributions = []

        for result in temporal_results:
            emotions = result.get(
                "emotions"
            )

            if (
                isinstance(emotions, dict)
                and emotions
            ):
                emotion_distributions.append(
                    emotions
                )

        if not emotion_distributions:
            raise ValueError(
                "Temporal predictions do not contain "
                "emotion distributions."
            )

        aggregator = EmotionAggregator()

        aggregated = aggregator.aggregate(
            emotion_distributions
        )

        message = (
            "Emotion estimated using a pretrained "
            "facial-expression recognition model "
            "across temporal video windows. The "
            "result represents visible facial "
            "expression and may not fully represent "
            "the underlying emotion."
        )

        return {
            "status": "success",
            "message": message,
            "emotions": aggregated[
                "emotions"
            ],
            "dominant_emotion": aggregated[
                "dominant_emotion"
            ],
            "confidence": aggregated[
                "confidence"
            ],
        }


def create_video_predictor(
    model: Optional[VideoModelInterface] = None,
    **kwargs: Any,
) -> VideoPredictor:
    """Create a configured VideoPredictor instance."""

    return VideoPredictor(
        model=model,
        **kwargs,
    )