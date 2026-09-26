from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from ..config.config import EMOTION_LABELS


class EmotionAggregator:
    """Aggregate frame or clip-level emotion predictions."""

    def __init__(
        self,
        labels: Iterable[str] | None = None,
    ) -> None:
        self.labels = tuple(
            labels if labels is not None else EMOTION_LABELS
        )

        if not self.labels:
            raise ValueError(
                "At least one emotion label is required."
            )

        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                "Emotion labels must be unique."
            )

    def aggregate(
        self,
        predictions: Iterable[Mapping[str, float]],
    ) -> Dict[str, object]:
        """Aggregate multiple emotion predictions.

        Each prediction is normalized before averaging so that
        small differences in probability scale do not affect
        the final result.

        Returns:
            Dictionary containing:

            - emotions: normalized emotion distribution
            - dominant_emotion: highest-probability emotion
            - confidence: probability of the dominant emotion
        """

        prediction_list = list(predictions)

        if not prediction_list:
            raise ValueError(
                "At least one prediction is required."
            )

        totals = {
            label: 0.0
            for label in self.labels
        }

        valid_prediction_count = 0

        for prediction_index, prediction in enumerate(
            prediction_list
        ):
            if not isinstance(prediction, Mapping):
                raise ValueError(
                    f"Prediction at index {prediction_index} "
                    "must be a mapping."
                )

            if not prediction:
                raise ValueError(
                    f"Prediction at index {prediction_index} "
                    "is empty."
                )

            unknown_labels = (
                set(prediction) - set(self.labels)
            )

            if unknown_labels:
                raise ValueError(
                    f"Prediction at index {prediction_index} "
                    f"contains unknown emotion labels: "
                    f"{sorted(unknown_labels)}"
                )

            prediction_values: Dict[str, float] = {}

            for label, value in prediction.items():
                value = float(value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"Probability for '{label}' "
                        "must be finite."
                    )

                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"Probability for '{label}' must be "
                        "between 0.0 and 1.0."
                    )

                prediction_values[label] = value

            total_probability = sum(
                prediction_values.values()
            )

            if total_probability <= 0.0:
                raise ValueError(
                    f"Prediction at index {prediction_index} "
                    "has zero probability mass."
                )

            # Normalize the individual prediction.
            for label in self.labels:
                value = prediction_values.get(
                    label,
                    0.0,
                )

                totals[label] += (
                    value / total_probability
                )

            valid_prediction_count += 1

        # Average the normalized predictions.
        emotions = {
            label: totals[label] / valid_prediction_count
            for label in self.labels
        }

        # Normalize the final distribution again to eliminate
        # floating-point accumulation errors.
        emotions = self._normalize_distribution(
            emotions
        )

        dominant_emotion = max(
            emotions,
            key=emotions.get,
        )

        confidence = emotions[dominant_emotion]

        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
        }

    def dominant_emotion(
        self,
        distribution: Mapping[str, float],
    ) -> tuple[str, float]:
        """Return the dominant emotion and its confidence."""

        normalized = self._normalize_distribution(
            distribution
        )

        emotion = max(
            normalized,
            key=normalized.get,
        )

        return (
            emotion,
            normalized[emotion],
        )

    def _normalize_distribution(
        self,
        distribution: Mapping[str, float],
    ) -> Dict[str, float]:
        """Validate and normalize an emotion distribution."""

        if not distribution:
            raise ValueError(
                "Emotion distribution cannot be empty."
            )

        unknown_labels = (
            set(distribution) - set(self.labels)
        )

        if unknown_labels:
            raise ValueError(
                "Distribution contains unknown "
                f"emotion labels: {sorted(unknown_labels)}"
            )

        normalized: Dict[str, float] = {}

        for label in self.labels:
            value = float(
                distribution.get(
                    label,
                    0.0,
                )
            )

            if not np.isfinite(value):
                raise ValueError(
                    f"Probability for '{label}' "
                    "must be finite."
                )

            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"Probability for '{label}' must be "
                    "between 0.0 and 1.0."
                )

            normalized[label] = value

        total = sum(
            normalized.values()
        )

        if total <= 0.0:
            raise ValueError(
                "Emotion distribution must contain "
                "positive probability mass."
            )

        return {
            label: value / total
            for label, value in normalized.items()
        }


def aggregate_emotion_distribution(
    predictions: Iterable[Mapping[str, float]],
) -> Dict[str, object]:
    """Convenience function for aggregating emotion predictions."""

    aggregator = EmotionAggregator()

    return aggregator.aggregate(
        predictions
    )