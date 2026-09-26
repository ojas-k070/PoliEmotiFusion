from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import cv2

from ..config.config import SUPPORTED_VIDEO_EXTENSIONS


@dataclass
class VideoMetadata:
    """Metadata extracted from a readable video file."""

    video_path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    is_valid: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert video metadata into a serializable dictionary."""

        return {
            "video_path": self.video_path,
            "fps": float(self.fps),
            "frame_count": int(self.frame_count),
            "duration_seconds": float(
                self.duration_seconds
            ),
            "width": int(self.width),
            "height": int(self.height),
            "is_valid": bool(self.is_valid),
            "error": self.error,
        }


class VideoReader:
    """Reusable helper for reading and validating video metadata."""

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the video reader.

        Extensions are normalized to lowercase so that files such as
        VIDEO.MP4 and video.mp4 are treated consistently.
        """

        extensions = (
            supported_extensions
            if supported_extensions is not None
            else SUPPORTED_VIDEO_EXTENSIONS
        )

        self.supported_extensions = {
            extension.lower()
            for extension in extensions
        }

    def _validate_path(
        self,
        video_path: str | Path,
    ) -> Path:
        """Validate the video path and supported extension."""

        path = Path(video_path)

        if not path.suffix:
            raise ValueError(
                "Video file must have a supported extension."
            )

        if (
            path.suffix.lower()
            not in self.supported_extensions
        ):
            raise ValueError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported: "
                f"{sorted(self.supported_extensions)}"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        return path

    def read_metadata(
        self,
        video_path: str | Path,
    ) -> VideoMetadata:
        """Open the video and return structured metadata.

        Args:
            video_path:
                Path to the video file.

        Returns:
            VideoMetadata containing video properties and validation
            status.

        Raises:
            ValueError:
                If the file extension is unsupported or the path is
                not a valid file.
            FileNotFoundError:
                If the video file does not exist.
        """

        path = self._validate_path(video_path)

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()

            return VideoMetadata(
                video_path=str(path),
                fps=0.0,
                frame_count=0,
                duration_seconds=0.0,
                width=0,
                height=0,
                is_valid=False,
                error=(
                    "Unable to open video file or "
                    "file is corrupted."
                ),
            )

        try:
            frame_count = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            # Validate all essential video metadata.
            if (
                frame_count <= 0
                or fps <= 0
                or width <= 0
                or height <= 0
            ):
                return VideoMetadata(
                    video_path=str(path),
                    fps=fps,
                    frame_count=frame_count,
                    duration_seconds=0.0,
                    width=width,
                    height=height,
                    is_valid=False,
                    error=(
                        "Video is empty or metadata "
                        "is invalid."
                    ),
                )

            duration_seconds = (
                frame_count / fps
            )

            return VideoMetadata(
                video_path=str(path),
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration_seconds,
                width=width,
                height=height,
                is_valid=True,
                error=None,
            )

        finally:
            capture.release()