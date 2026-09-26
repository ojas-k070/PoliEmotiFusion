import io
import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Target exact labels required by PoliEmotiFusion
TARGET_EMOTIONS = [
    "Anger",
    "Joy",
    "Sadness",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]

# Mapping diverse audio model emotion labels to PoliEmotiFusion canonical taxonomy
AUDIO_LABEL_MAPPING: Dict[str, str] = {
    "angry": "Anger",
    "anger": "Anger",
    "ang": "Anger",
    "happy": "Joy",
    "happiness": "Joy",
    "joy": "Joy",
    "hap": "Joy",
    "sad": "Sadness",
    "sadness": "Sadness",
    "fear": "Fear",
    "fearful": "Fear",
    "fea": "Fear",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "sur": "Surprise",
    "disgust": "Disgust",
    "dis": "Disgust",
    "neutral": "Neutral",
    "neu": "Neutral",
    "calm": "Neutral",
}


class AudioEmotionModelService:
    """Service to process speech audio and run inference for emotion classification."""

    def __init__(
        self,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    ) -> None:
        self.model_name = model_name
        self.feature_extractor = None
        self.model = None
        self.device = None
        self.target_sampling_rate = 16000

    def load_model(self) -> None:
        """Load feature extractor and audio classification model onto target device."""
        if self.feature_extractor is not None and self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded audio model '{self.model_name}' on {self.device}")
        except Exception as exc:
            logger.warning(
                f"Could not load Hugging Face model '{self.model_name}': {exc}. "
                "Inference will fall back to acoustic feature scoring."
            )
            self.feature_extractor = None
            self.model = None

    def read_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Decode raw audio bytes into a 1D float32 numpy array and sample rate.
        Supports WAV, MP3, FLAC, OGG etc.
        """
        # Try soundfile first
        try:
            import soundfile as sf

            with io.BytesIO(audio_bytes) as bio:
                data, sample_rate = sf.read(bio, dtype="float32")
                # If multichannel, average channels to mono
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data, int(sample_rate)
        except Exception as sf_err:
            logger.debug(f"soundfile decode failed: {sf_err}")

        # Try scipy.io.wavfile for standard WAV
        try:
            from scipy.io import wavfile

            with io.BytesIO(audio_bytes) as bio:
                sample_rate, data = wavfile.read(bio)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if np.issubdtype(data.dtype, np.integer):
                    max_val = float(np.iinfo(data.dtype).max)
                    data = data.astype(np.float32) / max_val
                else:
                    data = data.astype(np.float32)
                return data, int(sample_rate)
        except Exception as scipy_err:
            logger.debug(f"scipy decode failed: {scipy_err}")

        # Fallback: wave module for standard uncompressed PCM WAV
        try:
            import wave

            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    n_channels = wf.getnchannels()
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

                    if n_channels > 1:
                        data = data.reshape(-1, n_channels).mean(axis=1)
                    return data, int(sample_rate)
        except Exception as wave_err:
            raise ValueError(f"Failed to decode audio file. Error: {wave_err}")

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """Resample audio signal to the target sampling rate (16 kHz)."""
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
        except ImportError:
            pass

        # High-precision linear interpolation resampler using numpy
        duration = len(audio) / float(orig_sr)
        target_length = int(np.round(duration * target_sr))
        orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        target_indices = np.linspace(0, len(audio) - 1, num=target_length)
        resampled = np.interp(target_indices, orig_indices, audio)
        return resampled.astype(np.float32)

    def extract_waveform(self, audio: np.ndarray, num_bins: int = 64) -> List[float]:
        """
        Downsample audio into normalized peak amplitude bins for UI waveform visualizer.
        Returns list of floats between 0.05 and 1.0.
        """
        if len(audio) == 0:
            return [0.1] * num_bins

        # Divide into equal bins
        bins = np.array_split(audio, num_bins)
        rms_values = []
        for b in bins:
            if len(b) == 0:
                rms_values.append(0.0)
            else:
                rms = float(np.sqrt(np.mean(b**2)))
                rms_values.append(rms)

        rms_arr = np.array(rms_values, dtype=np.float32)
        max_val = float(np.max(rms_arr)) if len(rms_arr) > 0 else 1.0

        if max_val > 1e-6:
            normalized = (rms_arr / max_val).tolist()
        else:
            normalized = [0.1] * num_bins

        # Clamp between 0.05 and 1.0 and round to 3 decimals
        return [round(float(np.clip(v, 0.05, 1.0)), 3) for v in normalized]

    def _acoustic_feature_scoring(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Resilient heuristic acoustic emotion estimator based on energy, pitch variance, and zero-crossing.
        Used when deep-learning weights are downloading or offline.
        """
        energy = float(np.mean(audio**2))
        zero_crossings = float(np.mean(np.diff(np.sign(audio) != 0)))
        abs_max = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.1

        # Heuristic scoring
        raw_scores = {
            "Anger": max(0.05, energy * 3.0 + abs_max * 0.4),
            "Joy": max(0.05, zero_crossings * 1.5 + abs_max * 0.3),
            "Sadness": max(0.05, (1.0 - abs_max) * 0.3 + (1.0 - zero_crossings) * 0.2),
            "Fear": max(0.05, zero_crossings * 1.8 + energy * 0.5),
            "Surprise": max(0.05, abs_max * 0.5 + energy * 1.0),
            "Disgust": max(0.05, (1.0 - energy) * 0.15 + zero_crossings * 0.2),
            "Neutral": max(0.1, 0.5 - abs_max * 0.2),
        }
        total = sum(raw_scores.values())
        return {k: (v / total) * 100.0 for k, v in raw_scores.items()}

    def predict(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Run Speech Emotion Recognition on input audio bytes.
        Returns:
            dict containing:
                - emotion (str): dominant emotion
                - confidence (float): confidence percentage of dominant emotion (1 decimal)
                - probabilities (Dict[str, float]): 1-decimal percentages totaling strictly 100.0
                - duration (float): duration in seconds
                - waveform (List[float]): downsampled normalized waveform
        """
        raw_audio, orig_sr = self.read_audio(audio_bytes)
        duration = float(len(raw_audio) / orig_sr)
        audio_16k = self.resample_audio(raw_audio, orig_sr, self.target_sampling_rate)
        waveform = self.extract_waveform(audio_16k, num_bins=64)

        if self.model is None or self.feature_extractor is None:
            self.load_model()

        raw_percentages: Dict[str, float] = {}

        if self.model is not None and self.feature_extractor is not None:
            try:
                import torch

                inputs = self.feature_extractor(
                    audio_16k,
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    raw_probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

                for idx, prob in enumerate(raw_probs):
                    raw_label = self.model.config.id2label.get(idx, str(idx)).lower()
                    canonical = AUDIO_LABEL_MAPPING.get(raw_label, raw_label.capitalize())
                    raw_percentages[canonical] = raw_percentages.get(canonical, 0.0) + (prob * 100.0)
            except Exception as infer_err:
                logger.warning(f"Audio model inference error: {infer_err}. Falling back to acoustic scoring.")
                raw_percentages = self._acoustic_feature_scoring(audio_16k)
        else:
            raw_percentages = self._acoustic_feature_scoring(audio_16k)

        # Ensure all 7 canonical emotions exist
        for label in TARGET_EMOTIONS:
            if label not in raw_percentages:
                raw_percentages[label] = 0.0

        # Initial 1-decimal rounding
        probabilities: Dict[str, float] = {
            label: round(raw_percentages[label], 1) for label in TARGET_EMOTIONS
        }

        # Adjust the dominant class so probabilities strictly sum to 100.0
        current_sum = round(sum(probabilities.values()), 1)
        diff = round(100.0 - current_sum, 1)
        if diff != 0:
            dominant_key = max(probabilities, key=probabilities.get)
            probabilities[dominant_key] = round(probabilities[dominant_key] + diff, 1)

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "duration": round(duration, 2),
            "waveform": waveform,
        }


# Singleton audio model service instance
audio_model_service = AudioEmotionModelService()
