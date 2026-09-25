from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence


class VideoModelInterface(ABC):
    """Abstract interface for future video emotion models."""

    @abstractmethod
    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Load or initialize the underlying model."""

    @abstractmethod
    def predict(self, frames: Sequence[Any]) -> Any:
        """Return a prediction for a single video clip or frame sequence."""

    @abstractmethod
    def predict_batch(self, frames_batch: Iterable[Sequence[Any]]) -> List[Any]:
        """Return predictions for a batch of clips."""

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Return the model label list."""
