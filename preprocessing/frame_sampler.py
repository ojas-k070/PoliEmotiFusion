from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class FrameSampleConfig:
    """Configuration for sampling frames from a video."""

    sample_rate: int = 1
    max_frames: int = 32
    target_resolution: Tuple[int, int] = (224, 224)
    start_time_seconds: float = 0.0
    max_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if self.max_frames <= 0:
            raise ValueError(
                "max_frames must be greater than zero."
            )

        width, height = self.target_resolution

        if width <= 0 or height <= 0:
            raise ValueError(
                "target_resolution values must be greater than zero."
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


class FrameSampler:
    """Sample video frames across the full requested video interval."""

    def __init__(
        self,
        config: FrameSampleConfig | None = None,
        sample_rate: int = 1,
        max_frames: int = 32,
        target_resolution: Tuple[int, int] = (224, 224),
        start_time_seconds: float = 0.0,
        max_duration_seconds: float | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = FrameSampleConfig(
                sample_rate=sample_rate,
                max_frames=max_frames,
                target_resolution=target_resolution,
                start_time_seconds=start_time_seconds,
                max_duration_seconds=max_duration_seconds,
            )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def max_frames(self) -> int:
        return self.config.max_frames

    @property
    def target_resolution(self) -> Tuple[int, int]:
        return self.config.target_resolution

    @property
    def start_time_seconds(self) -> float:
        return self.config.start_time_seconds

    @property
    def max_duration_seconds(self) -> float | None:
        return self.config.max_duration_seconds

    def sample(
        self,
        video_path: str | Path,
    ) -> List[np.ndarray]:
        """Sample frames uniformly across the requested interval."""
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():
            raise ValueError(
                f"Could not open video file: {path}"
            )

        try:
            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            frame_count = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if fps <= 0:
                raise ValueError(
                    "Could not determine video FPS."
                )

            if frame_count <= 0:
                raise ValueError(
                    "Video contains no readable frames."
                )

            duration_seconds = (
                frame_count / fps
            )

            if self.start_time_seconds >= duration_seconds:
                raise ValueError(
                    "start_time_seconds is beyond "
                    "the end of the video."
                )

            start_time = self.start_time_seconds

            available_duration = (
                duration_seconds
                - start_time
            )

            if self.max_duration_seconds is not None:
                available_duration = min(
                    available_duration,
                    self.max_duration_seconds,
                )

            end_time = (
                start_time
                + available_duration
            )

            start_frame = max(
                0,
                int(
                    round(
                        start_time * fps
                    )
                ),
            )

            end_frame = min(
                frame_count - 1,
                int(
                    round(
                        end_time * fps
                    )
                ),
            )

            if end_frame < start_frame:
                end_frame = start_frame

            # First apply the requested sample rate.
            candidate_indices = np.arange(
                start_frame,
                end_frame + 1,
                self.sample_rate,
                dtype=np.int64,
            )

            if len(candidate_indices) == 0:
                candidate_indices = np.array(
                    [start_frame],
                    dtype=np.int64,
                )

            # If there are more candidate frames than the
            # configured maximum, distribute the selected
            # frames uniformly across the complete interval.
            if len(candidate_indices) > self.max_frames:
                selected_positions = np.linspace(
                    0,
                    len(candidate_indices) - 1,
                    self.max_frames,
                    dtype=np.int64,
                )

                frame_indices = candidate_indices[
                    selected_positions
                ]
            else:
                frame_indices = candidate_indices

            frames: List[np.ndarray] = []

            for frame_index in frame_indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index),
                )

                success, frame = capture.read()

                if not success or frame is None:
                    continue

                frames.append(
                    self._resize_frame(frame)
                )

            if not frames:
                raise ValueError(
                    "No frames could be extracted "
                    "from the video."
                )

            return frames

        finally:
            capture.release()

    def _resize_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Resize a frame to the configured resolution."""
        width, height = self.target_resolution

        return cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )


def sample_video_frames(
    video_path: str | Path,
    sample_rate: int = 1,
    max_frames: int = 32,
    target_resolution: Tuple[int, int] = (224, 224),
    start_time_seconds: float = 0.0,
    max_duration_seconds: float | None = None,
) -> List[np.ndarray]:
    """Convenience function for sampling video frames."""
    config = FrameSampleConfig(
        sample_rate=sample_rate,
        max_frames=max_frames,
        target_resolution=target_resolution,
        start_time_seconds=start_time_seconds,
        max_duration_seconds=max_duration_seconds,
    )

    sampler = FrameSampler(
        config=config
    )

    return sampler.sample(
        video_path
    )