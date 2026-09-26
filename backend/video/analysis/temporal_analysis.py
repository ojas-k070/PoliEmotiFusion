from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


@dataclass
class TemporalEmotionPoint:
    """Emotion prediction associated with a point in the video."""

    timestamp_seconds: float
    emotion: str
    confidence: float

    def __post_init__(self) -> None:
        if self.timestamp_seconds < 0:
            raise ValueError(
                "timestamp_seconds cannot be negative."
            )

        if not self.emotion:
            raise ValueError(
                "emotion cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the temporal point to a dictionary.

        Both ``timestamp`` and ``timestamp_seconds`` are provided
        for compatibility with the existing project interface.
        """

        return {
            "timestamp": float(
                self.timestamp_seconds
            ),
            "timestamp_seconds": float(
                self.timestamp_seconds
            ),
            "emotion": self.emotion,
            "confidence": float(
                self.confidence
            ),
        }


@dataclass
class TemporalEmotionTimeline:
    """Ordered collection of temporal emotion predictions."""

    points: List[TemporalEmotionPoint] = field(
        default_factory=list
    )

    def add_event(
        self,
        timestamp_seconds: float,
        emotion: str,
        confidence: float,
    ) -> None:
        """Add an emotion event to the timeline."""

        point = TemporalEmotionPoint(
            timestamp_seconds=timestamp_seconds,
            emotion=emotion,
            confidence=confidence,
        )

        self.points.append(point)

        # Keep events in chronological order.
        self.points.sort(
            key=lambda item: item.timestamp_seconds
        )

    def add_point(
        self,
        timestamp_seconds: float,
        emotion: str,
        confidence: float,
    ) -> None:
        """Add an emotion point.

        This is an alias for ``add_event`` kept for compatibility.
        """

        self.add_event(
            timestamp_seconds=timestamp_seconds,
            emotion=emotion,
            confidence=confidence,
        )

    def extend(
        self,
        points: Iterable[TemporalEmotionPoint],
    ) -> None:
        """Add multiple temporal emotion points."""

        for point in points:
            if not isinstance(
                point,
                TemporalEmotionPoint,
            ):
                raise ValueError(
                    "All timeline points must be "
                    "TemporalEmotionPoint instances."
                )

            self.points.append(point)

        self.points.sort(
            key=lambda item: item.timestamp_seconds
        )

    def to_list(self) -> List[Dict[str, Any]]:
        """Return the timeline as a list of dictionaries."""

        return [
            point.to_dict()
            for point in self.points
        ]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return the timeline as a list of dictionaries.

        Alias for ``to_list()``.
        """

        return self.to_list()

    def emotions(self) -> List[str]:
        """Return emotions in chronological order."""

        return [
            point.emotion
            for point in self.points
        ]

    def get_changes(self) -> List[Dict[str, Any]]:
        """Return points where the emotion changes."""

        if not self.points:
            return []

        changes: List[Dict[str, Any]] = []

        previous_emotion = self.points[0].emotion

        for point in self.points[1:]:
            if point.emotion != previous_emotion:
                changes.append(
                    {
                        "timestamp": float(
                            point.timestamp_seconds
                        ),
                        "timestamp_seconds": float(
                            point.timestamp_seconds
                        ),
                        "from_emotion": previous_emotion,
                        "to_emotion": point.emotion,
                        "confidence": float(
                            point.confidence
                        ),
                    }
                )

                previous_emotion = point.emotion

        return changes


def build_temporal_timeline(
    predictions: Iterable[Mapping[str, Any]],
) -> TemporalEmotionTimeline:
    """Build a temporal emotion timeline from predictions."""

    timeline = TemporalEmotionTimeline()

    for index, prediction in enumerate(
        predictions
    ):
        if not isinstance(
            prediction,
            Mapping,
        ):
            raise ValueError(
                f"Prediction at index {index} "
                "must be a mapping."
            )

        # Support both timestamp naming conventions.
        if "timestamp_seconds" in prediction:
            timestamp = prediction[
                "timestamp_seconds"
            ]
        elif "timestamp" in prediction:
            timestamp = prediction[
                "timestamp"
            ]
        else:
            raise ValueError(
                f"Prediction at index {index} "
                "is missing 'timestamp' or "
                "'timestamp_seconds'."
            )

        if "emotion" not in prediction:
            raise ValueError(
                f"Prediction at index {index} "
                "is missing 'emotion'."
            )

        if "confidence" not in prediction:
            raise ValueError(
                f"Prediction at index {index} "
                "is missing 'confidence'."
            )

        timeline.add_event(
            timestamp_seconds=float(timestamp),
            emotion=str(
                prediction["emotion"]
            ),
            confidence=float(
                prediction["confidence"]
            ),
        )

    return timeline


def aggregate_temporal_predictions(
    predictions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate temporal predictions into a timeline summary."""

    timeline = build_temporal_timeline(
        predictions
    )

    return {
        "timeline": timeline.to_list(),
        "changes": timeline.get_changes(),
        "num_points": len(
            timeline.points
        ),
    }


def analyze_temporal_changes(
    predictions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Analyze emotion changes across a video timeline."""

    return aggregate_temporal_predictions(
        predictions
    )