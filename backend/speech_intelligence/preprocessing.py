
import io
import logging
import subprocess
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Decode raw audio bytes into a 1D float32 numpy array,
    original sample rate, and number of channels.

    FFmpeg is used as the primary decoder so formats such as
    MP3, M4A, OGG, FLAC, and WAV are supported reliably.
    """

    # 1. FFmpeg - primary decoder
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if process.returncode != 0:
            error_message = process.stderr.decode(
                errors="replace"
            ).strip()

            raise ValueError(
                f"FFmpeg failed to decode audio: {error_message}"
            )

        if not process.stdout:
            raise ValueError(
                "FFmpeg returned empty audio data"
            )

        # FFmpeg converted the input into WAV.
        # Use Python's built-in wave module to read it.
        import wave

        with io.BytesIO(process.stdout) as bio:
            with wave.open(bio, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(
                    wf.getnframes()
                )

        # Convert PCM WAV data to float32
        if sample_width == 1:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.uint8,
                ).astype(np.float32)
                - 128.0
            ) / 128.0

        elif sample_width == 2:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.int16,
                ).astype(np.float32)
                / 32768.0
            )

        elif sample_width == 4:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.int32,
                ).astype(np.float32)
                / 2147483648.0
            )

        else:
            raise ValueError(
                f"Unsupported WAV sample width: {sample_width}"
            )

        # Convert stereo/multi-channel audio to mono
        if channels > 1:
            data = data.reshape(
                -1,
                channels,
            ).mean(axis=1)

        return (
            data.astype(np.float32),
            int(sample_rate),
            channels,
        )

    except FileNotFoundError:
        logger.warning(
            "FFmpeg executable was not found in PATH. "
            "Trying fallback decoders."
        )

    except Exception as ffmpeg_err:
        logger.warning(
            f"FFmpeg decoding failed: {ffmpeg_err}"
        )

    # 2. Try soundfile directly
    try:
        import soundfile as sf

        with io.BytesIO(audio_bytes) as bio:
            data, sample_rate = sf.read(
                bio,
                dtype="float32",
            )

        channels = data.shape[1] if data.ndim > 1 else 1

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        return (
            data.astype(np.float32),
            int(sample_rate),
            channels,
        )

    except Exception as sf_err:
        logger.debug(
            f"soundfile decode fallback: {sf_err}"
        )

    # 3. Try scipy WAV decoder
    try:
        from scipy.io import wavfile

        with io.BytesIO(audio_bytes) as bio:
            sample_rate, data = wavfile.read(bio)

        channels = data.shape[1] if data.ndim > 1 else 1

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if np.issubdtype(data.dtype, np.integer):
            max_val = float(
                np.iinfo(data.dtype).max
            )

            if max_val > 0:
                data = (
                    data.astype(np.float32)
                    / max_val
                )
            else:
                data = data.astype(np.float32)
        else:
            data = data.astype(np.float32)

        return (
            data,
            int(sample_rate),
            channels,
        )

    except Exception as scipy_err:
        logger.debug(
            f"scipy decode fallback: {scipy_err}"
        )

    # 4. Final WAV decoder fallback
    try:
        import wave

        with io.BytesIO(audio_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(
                    wf.getnframes()
                )

        if sample_width == 2:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.int16,
                ).astype(np.float32)
                / 32768.0
            )

        elif sample_width == 4:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.int32,
                ).astype(np.float32)
                / 2147483648.0
            )

        elif sample_width == 1:
            data = (
                np.frombuffer(
                    frames,
                    dtype=np.uint8,
                ).astype(np.float32)
                - 128.0
            ) / 128.0

        else:
            raise ValueError(
                f"Unsupported sample width: {sample_width}"
            )

        if channels > 1:
            data = data.reshape(
                -1,
                channels,
            ).mean(axis=1)

        return (
            data.astype(np.float32),
            int(sample_rate),
            channels,
        )

    except Exception as wave_err:
        raise ValueError(
            f"Unable to decode audio stream: {wave_err}"
        )

def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample 1D float32 audio to target sample rate.
    Default target is 16000 Hz.
    """

    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        import librosa

        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
        ).astype(np.float32)

    except ImportError:
        pass

    # Linear interpolation fallback
    duration = len(audio) / float(orig_sr)

    target_length = int(
        np.round(duration * target_sr)
    )

    if target_length <= 0:
        return np.array([], dtype=np.float32)

    orig_indices = np.linspace(
        0,
        len(audio) - 1,
        num=len(audio),
    )

    target_indices = np.linspace(
        0,
        len(audio) - 1,
        num=target_length,
    )

    resampled = np.interp(
        target_indices,
        orig_indices,
        audio,
    )

    return resampled.astype(np.float32)


def normalize_volume(
    audio: np.ndarray,
    target_peak: float = 0.95,
) -> np.ndarray:
    """
    Normalize audio amplitude to prevent clipping
    and improve faint speech.
    """

    if len(audio) == 0:
        return audio

    peak = float(
        np.max(np.abs(audio))
    )

    if peak > 1e-4:
        gain = target_peak / peak

        # Prevent extreme amplification of noise
        gain = min(gain, 10.0)

        return (
            audio * gain
        ).astype(np.float32)

    return audio


def trim_silence(
    audio: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -45.0,
) -> np.ndarray:
    """
    Trim leading and trailing silence below
    the specified dB threshold.
    """

    if len(audio) == 0:
        return audio

    # 20 ms frames
    frame_len = int(sr * 0.02)

    if len(audio) < frame_len * 2:
        return audio

    num_frames = len(audio) // frame_len

    frames = audio[
        : num_frames * frame_len
    ].reshape(
        num_frames,
        frame_len,
    )

    frame_rms = np.sqrt(
        np.mean(frames**2, axis=1)
        + 1e-12
    )

    frame_db = 20 * np.log10(
        frame_rms + 1e-12
    )

    active_indices = np.where(
        frame_db > threshold_db
    )[0]

    if len(active_indices) == 0:
        return audio

    start_sample = max(
        0,
        int(
            (active_indices[0] - 1)
            * frame_len
        ),
    )

    end_sample = min(
        len(audio),
        int(
            (active_indices[-1] + 2)
            * frame_len
        ),
    )

    trimmed = audio[
        start_sample:end_sample
    ]

    # Ensure at least 0.5 seconds remain
    if len(trimmed) >= int(sr * 0.5):
        return trimmed.astype(np.float32)

    return audio


def chunk_audio(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_duration_sec: float = 25.0,
    overlap_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Split long speech audio into sequential
    chunks of approximately 20-30 seconds.

    Returns a list of dictionaries containing:

        index
        audio
        start_time
        end_time
        duration
    """

    total_samples = len(audio)

    chunk_samples = int(
        chunk_duration_sec * sr
    )

    overlap_samples = int(
        overlap_sec * sr
    )

    step_samples = (
        chunk_samples - overlap_samples
    )

    if total_samples <= chunk_samples:
        return [
            {
                "index": 0,
                "audio": audio.astype(
                    np.float32
                ),
                "start_time": 0.0,
                "end_time": round(
                    total_samples / sr,
                    2,
                ),
                "duration": round(
                    total_samples / sr,
                    2,
                ),
            }
        ]

    chunks = []

    idx = 0
    start = 0

    while start < total_samples:

        end = min(
            start + chunk_samples,
            total_samples,
        )

        segment = audio[start:end]

        # Ignore tiny leftover tail
        if (
            len(segment)
            < int(0.5 * sr)
            and chunks
        ):
            break

        start_time = round(
            start / sr,
            2,
        )

        end_time = round(
            end / sr,
            2,
        )

        duration = round(
            (end - start) / sr,
            2,
        )

        chunks.append(
            {
                "index": idx,
                "audio": segment.astype(
                    np.float32
                ),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
            }
        )

        idx += 1
        start += step_samples

    return chunks


def preprocess_speech_audio(
    audio_bytes: bytes,
    filename: str = "audio_recording",
    file_size_formatted: str = "",
) -> Dict[str, Any]:
    """
    Full audio preprocessing pipeline:

    1. Decode multi-format audio using FFmpeg
    2. Resample to 16 kHz
    3. Convert to mono
    4. Trim leading/trailing silence
    5. Normalize volume
    6. Partition long recordings into 25-second chunks
    """

    # Decode audio
    raw_audio, orig_sr, channels = decode_audio_bytes(
        audio_bytes
    )

    orig_duration = float(
        len(raw_audio) / orig_sr
    )

    # Resample to 16 kHz
    audio_16k = resample_audio(
        raw_audio,
        orig_sr,
        target_sr=16000,
    )

    # Trim leading/trailing silence
    trimmed_audio = trim_silence(
        audio_16k,
        sr=16000,
    )

    # Normalize volume
    normalized_audio = normalize_volume(
        trimmed_audio
    )

    # Split into 25-second chunks
    chunks = chunk_audio(
        normalized_audio,
        sr=16000,
        chunk_duration_sec=25.0,
    )

    # File extension
    ext = (
        filename.split(".")[-1].upper()
        if "." in filename
        else "AUDIO"
    )

    # Channel description
    channel_desc = (
        "Mono"
        if channels == 1
        else "Stereo"
        if channels == 2
        else f"{channels} channels"
    )

    stats = {
        "duration": round(
            orig_duration,
            2,
        ),
        "processedDuration": round(
            len(normalized_audio)
            / 16000.0,
            2,
        ),
        "sampleRate": orig_sr,
        "targetSampleRate": 16000,
        "channels": channels,
        "channelDesc": channel_desc,
        "format": ext,
        "fileSize": file_size_formatted,
        "chunkCount": len(chunks),
    }

    logger.info(
        f"Preprocessed audio '{filename}': "
        f"orig_sr={orig_sr}, "
        f"duration={orig_duration:.2f}s, "
        f"chunks={len(chunks)}, "
        f"channels={channel_desc}"
    )

    return {
        "full_audio": normalized_audio,
        "chunks": chunks,
        "stats": stats,
    }