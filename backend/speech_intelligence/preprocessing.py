import io
import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Decode raw audio bytes into a 1D float32 numpy array, original sample rate, and channels.
    Supports WAV, MP3, FLAC, M4A, OGG using soundfile, scipy, or wave.
    """
    # 1. Try soundfile
    try:
        import soundfile as sf

        with io.BytesIO(audio_bytes) as bio:
            data, sample_rate = sf.read(bio, dtype="float32")
            channels = data.shape[1] if data.ndim > 1 else 1
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sample_rate), channels
    except Exception as sf_err:
        logger.debug(f"soundfile decode fallback: {sf_err}")

    # 2. Try scipy.io.wavfile
    try:
        from scipy.io import wavfile

        with io.BytesIO(audio_bytes) as bio:
            sample_rate, data = wavfile.read(bio)
            channels = data.shape[1] if data.ndim > 1 else 1
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
            return data, int(sample_rate), channels
    except Exception as scipy_err:
        logger.debug(f"scipy decode fallback: {scipy_err}")

    # 3. Try standard wave module
    try:
        import wave

        with io.BytesIO(audio_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

                if sampwidth == 2:
                    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sampwidth == 1:
                    data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                if channels > 1:
                    data = data.reshape(-1, channels).mean(axis=1)
                return data.astype(np.float32), int(sample_rate), channels
    except Exception as wave_err:
        raise ValueError(f"Unable to decode audio stream: {wave_err}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D float32 audio to target sample rate (16000 Hz)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        pass

    # High-fidelity linear interpolation fallback
    duration = len(audio) / float(orig_sr)
    target_length = int(np.round(duration * target_sr))
    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=target_length)
    resampled = np.interp(target_indices, orig_indices, audio)
    return resampled.astype(np.float32)


def normalize_volume(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio amplitude to prevent clipping and enhance faint speech."""
    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-4:
        gain = target_peak / peak
        # Limit extreme amplification for noise
        gain = min(gain, 10.0)
        return (audio * gain).astype(np.float32)
    return audio


def trim_silence(audio: np.ndarray, sr: int = 16000, threshold_db: float = -45.0) -> np.ndarray:
    """Trim leading and trailing silence below dB threshold to focus on spoken discourse."""
    if len(audio) == 0:
        return audio

    # Compute energy in 20ms frames
    frame_len = int(sr * 0.02)
    if len(audio) < frame_len * 2:
        return audio

    num_frames = len(audio) // frame_len
    frames = audio[: num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    frame_db = 20 * np.log10(frame_rms + 1e-12)

    active_indices = np.where(frame_db > threshold_db)[0]
    if len(active_indices) == 0:
        return audio

    start_sample = max(0, int((active_indices[0] - 1) * frame_len))
    end_sample = min(len(audio), int((active_indices[-1] + 2) * frame_len))

    trimmed = audio[start_sample:end_sample]
    # Ensure minimum 0.5 second audio remains
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
    Split long speech audio into sequential chunks of 20-30 seconds.
    Whisper natively operates on 30-second receptive fields.
    Returns:
        List of chunk dicts: {
            "index": int,
            "audio": np.ndarray (float32),
            "start_time": float,
            "end_time": float,
            "duration": float
        }
    """
    total_samples = len(audio)
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = chunk_samples - overlap_samples

    if total_samples <= chunk_samples:
        return [
            {
                "index": 0,
                "audio": audio.astype(np.float32),
                "start_time": 0.0,
                "end_time": round(total_samples / sr, 2),
                "duration": round(total_samples / sr, 2),
            }
        ]

    chunks = []
    idx = 0
    start = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        segment = audio[start:end]

        # Ignore tiny leftover tail (<0.5 second) by appending to previous if available
        if len(segment) < int(0.5 * sr) and chunks:
            break

        start_time = round(start / sr, 2)
        end_time = round(end / sr, 2)
        duration = round((end - start) / sr, 2)

        chunks.append(
            {
                "index": idx,
                "audio": segment.astype(np.float32),
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
    1. Decode multi-format stream
    2. Resample to 16 kHz mono
    3. Trim dead padding silence
    4. Normalize volume
    5. Partition long recordings into 20-30s sequential chunks
    """
    raw_audio, orig_sr, channels = decode_audio_bytes(audio_bytes)
    orig_duration = float(len(raw_audio) / orig_sr)

    # Resample to 16kHz
    audio_16k = resample_audio(raw_audio, orig_sr, target_sr=16000)

    # Trim leading/trailing dead silence
    trimmed_audio = trim_silence(audio_16k, sr=16000)

    # Normalize volume
    normalized_audio = normalize_volume(trimmed_audio)

    # Partition into 20-30s chunks
    chunks = chunk_audio(normalized_audio, sr=16000, chunk_duration_sec=25.0)

    ext = filename.split(".")[-1].upper() if "." in filename else "AUDIO"
    channel_desc = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels} channels"

    stats = {
        "duration": round(orig_duration, 2),
        "processedDuration": round(len(normalized_audio) / 16000.0, 2),
        "sampleRate": orig_sr,
        "targetSampleRate": 16000,
        "channels": channels,
        "channelDesc": channel_desc,
        "format": ext,
        "fileSize": file_size_formatted,
        "chunkCount": len(chunks),
    }

    logger.info(
        f"Preprocessed audio '{filename}': orig_sr={orig_sr}, duration={orig_duration:.2f}s, "
        f"chunks={len(chunks)}, channels={channel_desc}"
    )

    return {
        "full_audio": normalized_audio,
        "chunks": chunks,
        "stats": stats,
    }
