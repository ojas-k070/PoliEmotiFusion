from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.config import EMOTION_LABELS


DATASET_METADATA_FIELDS: List[str] = [
    "video_id",
    "title",
    "source",
    "source_url",
    "category",
    "duration",
    "publication_date",
    "local_path",
    "emotion_label",
    "split",
]


VALID_DATASET_SPLITS = {
    "train",
    "validation",
    "test",
}


@dataclass
class DatasetMetadataSchema:
    """Metadata schema for the Indian political video dataset.

    The dataset keeps event/context information (`category`) separate
    from the emotion annotation (`emotion_label`).

    Emotion labels are intentionally optional until the video has
    actually been annotated.
    """

    video_id: str = ""
    title: str = ""
    source: str = ""
    source_url: str = ""
    category: str = "speech"
    duration: float = 0.0
    publication_date: str = ""
    local_path: str = ""
    emotion_label: Optional[str] = None
    split: str = "train"

    def __post_init__(self) -> None:
        """Validate dataset metadata."""

        if self.duration < 0:
            raise ValueError(
                "duration cannot be negative."
            )

        if self.split not in VALID_DATASET_SPLITS:
            raise ValueError(
                f"split must be one of "
                f"{sorted(VALID_DATASET_SPLITS)}."
            )

        if self.emotion_label is not None:
            if self.emotion_label not in EMOTION_LABELS:
                raise ValueError(
                    f"emotion_label must be one of "
                    f"{list(EMOTION_LABELS)} or None."
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata into a dictionary."""

        return {
            "video_id": self.video_id,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "category": self.category,
            "duration": float(self.duration),
            "publication_date": self.publication_date,
            "local_path": self.local_path,
            "emotion_label": self.emotion_label,
            "split": self.split,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "DatasetMetadataSchema":
        """Create a metadata record from a dictionary."""

        return cls(
            video_id=str(
                data.get("video_id", "")
            ),
            title=str(
                data.get("title", "")
            ),
            source=str(
                data.get("source", "")
            ),
            source_url=str(
                data.get("source_url", "")
            ),
            category=str(
                data.get("category", "speech")
            ),
            duration=float(
                data.get("duration", 0.0)
            ),
            publication_date=str(
                data.get("publication_date", "")
            ),
            local_path=str(
                data.get("local_path", "")
            ),
            emotion_label=data.get(
                "emotion_label"
            ),
            split=str(
                data.get("split", "train")
            ),
        )