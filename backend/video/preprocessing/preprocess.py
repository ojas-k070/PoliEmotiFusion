from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple
from urllib.request import urlopen

import cv2
import numpy as np

from .frame_sampler import FrameSampleConfig, FrameSampler


_YUNET_MODEL_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx"
)
_YUNET_CACHE_DIR = Path.home() / ".cache" / "politemotifusion"
_YUNET_MODEL_PATH = _YUNET_CACHE_DIR / "face_detection_yunet_2023mar.onnx"


def _ensure_yunet_model() -> Path:
    """Return a local YuNet model path, downloading it once if necessary."""
    if _YUNET_MODEL_PATH.exists() and _YUNET_MODEL_PATH.stat().st_size > 100_000:
        return _YUNET_MODEL_PATH

    _YUNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = _YUNET_MODEL_PATH.with_suffix(".download")

    try:
        with urlopen(_YUNET_MODEL_URL, timeout=60) as response:
            data = response.read()
        if len(data) < 100_000:
            raise RuntimeError("Downloaded YuNet model appears to be incomplete.")
        temporary_path.write_bytes(data)
        temporary_path.replace(_YUNET_MODEL_PATH)
    except Exception as exc:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the YuNet face-detection model. "
            f"Please check your internet connection and try again. Details: {exc}"
        ) from exc

    return _YUNET_MODEL_PATH


def _create_face_detector(
    *,
    input_size: tuple[int, int] = (320, 320),
    score_threshold: float = 0.55,
    nms_threshold: float = 0.30,
) -> cv2.FaceDetectorYN:
    """Create the OpenCV YuNet face detector."""
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "This OpenCV installation does not provide FaceDetectorYN. "
            "Please reinstall a compatible OpenCV 5.x package."
        )

    model_path = _ensure_yunet_model()

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        input_size,
        score_threshold,
        nms_threshold,
        5000,
        0,
        0,
    )

    if detector is None:
        raise RuntimeError("OpenCV YuNet face detector could not be created.")

    return detector


def _detect_and_crop_primary_face(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
    *,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    """Detect faces and return a padded crop of the largest detected face.

    The largest face is used to preserve the existing model input shape and
    temporal pipeline. If no face is detected, the original frame is returned
    so that a short detection failure does not crash the video analysis.
    """
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        return frame

    frame_height, frame_width = frame.shape[:2]

    if frame.shape[2] == 4:
        image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        image = frame

    detector.setInputSize((frame_width, frame_height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return frame

    # YuNet returns [x, y, width, height, landmarks..., score].
    primary = max(
        faces,
        key=lambda detection: float(detection[2]) * float(detection[3]),
    )

    x, y, width, height = [float(value) for value in primary[:4]]

    if width <= 0 or height <= 0:
        return frame

    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)

    left = max(0, int(x) - pad_x)
    top = max(0, int(y) - pad_y)
    right = min(frame_width, int(x + width) + pad_x)
    bottom = min(frame_height, int(y + height) + pad_y)

    crop = frame[top:bottom, left:right]

    if crop.size == 0:
        return frame

    return crop


def preprocess_frame_sequence(
    frames: Sequence[np.ndarray],
    *,
    resize: Tuple[int, int] | None = None,
    normalize: bool = True,
    detect_faces: bool = True,
    face_padding_ratio: float = 0.15,
    convert_bgr_to_rgb: bool = True,
) -> np.ndarray:
    """Prepare a sequence of video frames for model processing.

    When face detection is enabled, YuNet detects the largest visible face in
    each frame and crops it before resizing. Frames without a detected face
    fall back to the original frame.

    OpenCV video frames are BGR by default, while the FER model is trained on
    standard RGB image inputs. The conversion is therefore performed after
    face cropping and before model inference.
    """
    if not frames:
        raise ValueError("At least one frame is required for preprocessing.")

    if resize is not None:
        if len(resize) != 2 or resize[0] < 1 or resize[1] < 1:
            raise ValueError(
                "resize must contain two positive values: (height, width)."
            )

    if face_padding_ratio < 0:
        raise ValueError("face_padding_ratio must be non-negative.")

    detector = _create_face_detector() if detect_faces else None

    prepared: list[np.ndarray] = []
    expected_channels: int | None = None

    for index, frame in enumerate(frames):
        array = np.asarray(frame)

        if array.ndim != 3:
            raise ValueError(
                f"Frame {index} must have shape (height, width, channels); "
                f"got {array.shape}."
            )

        if array.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Frame {index} must have 1, 3, or 4 channels; "
                f"got {array.shape[2]}."
            )

        if expected_channels is None:
            expected_channels = array.shape[2]
        elif array.shape[2] != expected_channels:
            raise ValueError(
                "All frames must have the same number of channels."
            )

        if detect_faces and detector is not None and array.shape[2] in (3, 4):
            array = _detect_and_crop_primary_face(
                array,
                detector,
                padding_ratio=face_padding_ratio,
            )

        if resize is not None:
            target_height, target_width = resize

            interpolation = (
                cv2.INTER_AREA
                if array.shape[0] > target_height
                or array.shape[1] > target_width
                else cv2.INTER_LINEAR
            )

            array = cv2.resize(
                array,
                (target_width, target_height),
                interpolation=interpolation,
            )

        # VideoReader/OpenCV supplies BGR frames. The FER model was trained
        # with normal RGB image ordering, so convert only after face cropping
        # and resizing. Keep grayscale frames unchanged.
        if convert_bgr_to_rgb and array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        elif convert_bgr_to_rgb and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)

        prepared.append(array)

    sequence = np.stack(prepared, axis=0)

    if normalize:
        sequence = sequence.astype(np.float32, copy=False)
        if np.issubdtype(sequence.dtype, np.floating):
            sequence /= 255.0

    return sequence


def preprocess_video_frames(
    video_path: str | Path,
    *,
    sample_rate: int = 1,
    max_frames: int | None = 32,
    max_duration_seconds: float | None = None,
    target_resolution: Tuple[int, int] = (224, 224),
    start_time_seconds: float = 0.0,
    normalize: bool = True,
    detect_faces: bool = True,
    face_padding_ratio: float = 0.15,
    convert_bgr_to_rgb: bool = True,
) -> np.ndarray:
    """Sample video frames and prepare them for facial-expression inference."""
    config = FrameSampleConfig(
        sample_rate=sample_rate,
        max_frames=max_frames,
        max_duration_seconds=max_duration_seconds,
        target_resolution=target_resolution,
        start_time_seconds=start_time_seconds,
    )

    sampler = FrameSampler(config=config)
    frames = sampler.sample(video_path)

    return preprocess_frame_sequence(
        frames,
        resize=target_resolution,
        normalize=normalize,
        detect_faces=detect_faces,
        face_padding_ratio=face_padding_ratio,
        convert_bgr_to_rgb=convert_bgr_to_rgb,
    )
