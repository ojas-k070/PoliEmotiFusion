from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence


class VideoModelInterface(ABC):
    """Abstract interface for video emotion analysis models.

    Concrete video models must implement model loading, single-clip
    prediction, batch prediction, and label retrieval.

    The interface is intentionally independent of any specific model
    architecture so the current pretrained facial-expression model can
    coexist with a future VideoMAE-based implementation.
    """

    @abstractmethod
    def load_model(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Load or initialize the underlying video model."""

    @abstractmethod
    def predict(
        self,
        frames: Sequence[Any],
    ) -> Any:
        """Return an emotion prediction for a frame sequence."""

    @abstractmethod
    def predict_batch(
        self,
        frames_batch: Iterable[Sequence[Any]],
    ) -> List[Any]:
        """Return predictions for multiple video clips."""

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Return the emotion labels supported by the model."""