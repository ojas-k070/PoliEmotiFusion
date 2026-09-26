from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from .frame_sampler import FrameSampleConfig, FrameSampler


def preprocess_frame_sequence(
    frames: Sequence[np.ndarray],
    *,
    resize: Tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Prepare a sequence of video frames for model processing.

    Args:
        frames:
            Sequence of video frames represented as NumPy arrays.

        resize:
            Optional target resolution as (height, width).

        normalize:
            If True, convert pixel values from [0, 255] to
            float32 values in [0.0, 1.0].

    Returns:
        NumPy array with shape (T, H, W, C).

    Raises:
        ValueError:
            If no frames are provided, a frame is invalid, frame
            dimensions are inconsistent, or resize values are invalid.
    """

    if not frames:
        raise ValueError(
            "At least one frame is required for preprocessing."
        )

    if resize is not None:
        if (
            len(resize) != 2
            or resize[0] < 1
            or resize[1] < 1
        ):
            raise ValueError(
                "resize must contain two positive values: "
                "(height, width)."
            )

    prepared: list[np.ndarray] = []

    expected_channels: int | None = None

    for index, frame in enumerate(frames):
        array = np.asarray(frame)

        if array.size == 0:
            raise ValueError(
                f"Frame at index {index} is empty."
            )

        if array.ndim != 3:
            raise ValueError(
                f"Frame at index {index} must have shape "
                "(H, W, C), but received shape {array.shape}."
            )

        channels = array.shape[2]

        if channels not in (1, 3, 4):
            raise ValueError(
                f"Frame at index {index} has {channels} channels. "
                "Expected 1, 3, or 4 channels."
            )

        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise ValueError(
                "All frames must have the same number of channels."
            )

        if resize is not None:
            target_height, target_width = resize

            if (
                array.shape[0] != target_height
                or array.shape[1] != target_width
            ):
                array = cv2.resize(
                    array,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

        prepared.append(array)

    sequence = np.stack(
        prepared,
        axis=0,
    )

    if normalize:
        sequence = sequence.astype(
            np.float32,
            copy=False,
        )

        # Standard video pixel normalization.
        if np.issubdtype(
            sequence.dtype,
            np.floating,
        ):
            sequence /= 255.0

    return sequence


def preprocess_video_frames(
    video_path: str | Path,
    *,
    sample_rate: int = 1,
    max_frames: int = 32,
    max_duration_seconds: float | None = None,
    target_resolution: Tuple[int, int] = (224, 224),
    start_time_seconds: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Sample and preprocess frames from a video.

    Args:
        video_path:
            Path to the input video.

        sample_rate:
            Number of frames between sampled frames.

        max_frames:
            Maximum number of frames to return.

        max_duration_seconds:
            Optional maximum duration to process.

        target_resolution:
            Output resolution as (height, width).

        start_time_seconds:
            Timestamp at which sampling begins.

        normalize:
            If True, return float32 values in [0.0, 1.0].

    Returns:
        NumPy array with shape (T, H, W, C).
    """

    config = FrameSampleConfig(
        sample_rate=sample_rate,
        max_frames=max_frames,
        max_duration_seconds=max_duration_seconds,
        target_resolution=target_resolution,
        start_time_seconds=start_time_seconds,
    )

    sampler = FrameSampler(
        config=config,
    )

    frames = sampler.sample(video_path)

    # FrameSampler already resizes frames to target_resolution.
    # Therefore no second resize is necessary here.
    return preprocess_frame_sequence(
        frames,
        resize=None,
        normalize=normalize,
    )