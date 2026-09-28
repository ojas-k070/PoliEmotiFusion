import io
import logging
import threading
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Target canonical emotion taxonomy required by PoliEmotiFusion
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

# ISO code to human-readable language names
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
}


class SpeechEmotionRecognitionService:
    """
    Independent modular service for Speech Emotion Recognition (SER).
    Easily replaceable in the future (e.g. with fine-tuned Hindi/Indian SER models).
    """

    def __init__(
        self,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    ) -> None:
        self.model_name = model_name
        self.feature_extractor = None
        self.model = None
        self.device = None
        self.target_sampling_rate = 16000
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        """Check if feature extractor and SER model are loaded into memory."""
        return self.feature_extractor is not None and self.model is not None

    def load_model(self) -> None:
        """Load feature extractor and audio classification model onto target device safely."""
        if self.is_loaded():
            return

        with self._lock:
            if self.is_loaded():
                return

            try:
                import torch
                from transformers import (
                    AutoConfig,
                    AutoModelForAudioClassification,
                    Wav2Vec2FeatureExtractor,
                )

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)

                # Ensure classifier projection matches checkpoint dimensions (1024)
                cfg = AutoConfig.from_pretrained(self.model_name)
                cfg.classifier_proj_size = 1024
                self.model = AutoModelForAudioClassification.from_pretrained(
                    self.model_name,
                    config=cfg,
                )

                # In transformers >= 5.x, Wav2Vec2ForSequenceClassification expects
                # projector.* and classifier.* instead of legacy classifier.dense.* and classifier.output.*.
                # Remap checkpoint weights if needed using safetensors.
                try:
                    from safetensors import safe_open
                    from huggingface_hub import hf_hub_download

                    st_path = hf_hub_download(self.model_name, "model.safetensors")
                    with safe_open(st_path, framework="pt", device="cpu") as sf:
                        sf_keys = set(sf.keys())
                        if hasattr(self.model, "projector") and "classifier.dense.weight" in sf_keys:
                            self.model.projector.weight.data.copy_(sf.get_tensor("classifier.dense.weight"))
                            self.model.projector.bias.data.copy_(sf.get_tensor("classifier.dense.bias"))
                        if hasattr(self.model, "classifier") and "classifier.output.weight" in sf_keys:
                            self.model.classifier.weight.data.copy_(sf.get_tensor("classifier.output.weight"))
                            self.model.classifier.bias.data.copy_(sf.get_tensor("classifier.output.bias"))
                except Exception as remap_err:
                    logger.debug(f"SER head weight remapping notice: {remap_err}")

                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded SER model '{self.model_name}' on {self.device}")
            except Exception as exc:
                logger.warning(
                    f"Could not load Hugging Face SER model '{self.model_name}': {exc}. "
                    "Inference will fall back to acoustic feature scoring."
                )
                self.feature_extractor = None
                self.model = None

    def _acoustic_feature_scoring(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Resilient heuristic acoustic emotion estimator based on energy, pitch variance, and zero-crossing.
        Used when deep-learning weights are downloading or offline.
        """
        energy = float(np.mean(audio**2))
        zero_crossings = float(np.mean(np.diff(np.sign(audio) != 0)))
        abs_max = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.1

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

    def predict_emotion(self, audio_16k: np.ndarray, duration: float) -> Dict[str, Any]:
        """
        Run inference using the SER model on 16kHz audio array.
        Uses windowed segment sampling for long audio to avoid quadratic attention latency.
        """
        if not self.is_loaded():
            self.load_model()

        raw_percentages: Dict[str, float] = {}

        if self.model is not None and self.feature_extractor is not None:
            try:
                import torch

                window_samples = 6 * self.target_sampling_rate  # 6s window
                if len(audio_16k) > window_samples:
                    num_windows = min(8, max(3, int(duration // 10)))
                    start_indices = np.linspace(0, len(audio_16k) - window_samples, num=num_windows, dtype=int)
                    audio_segments = [audio_16k[s : s + window_samples] for s in start_indices]
                else:
                    audio_segments = [audio_16k]

                inputs = self.feature_extractor(
                    audio_segments,
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs_batch = torch.softmax(outputs.logits, dim=-1)
                    raw_probs = torch.mean(probs_batch, dim=0).cpu().tolist()

                for idx, prob in enumerate(raw_probs):
                    raw_label = self.model.config.id2label.get(idx, str(idx)).lower()
                    canonical = AUDIO_LABEL_MAPPING.get(raw_label, raw_label.capitalize())
                    raw_percentages[canonical] = raw_percentages.get(canonical, 0.0) + (prob * 100.0)
            except Exception as infer_err:
                logger.warning(f"SER inference error: {infer_err}. Using acoustic heuristic fallback.")
                raw_percentages = self._acoustic_feature_scoring(audio_16k)
        else:
            raw_percentages = self._acoustic_feature_scoring(audio_16k)

        # Guarantee all 7 canonical emotions exist
        for label in TARGET_EMOTIONS:
            if label not in raw_percentages:
                raw_percentages[label] = 0.0

        # 1-decimal percentages
        probabilities: Dict[str, float] = {
            label: round(raw_percentages[label], 1) for label in TARGET_EMOTIONS
        }

        # Normalize dominant class to strictly total 100.0%
        current_sum = round(sum(probabilities.values()), 1)
        diff = round(100.0 - current_sum, 1)
        if diff != 0:
            dominant_key = max(probabilities, key=probabilities.get)
            probabilities[dominant_key] = round(probabilities[dominant_key] + diff, 1)

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        # Build top-3 emotions
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [
            {"emotion": em, "confidence": conf}
            for em, conf in sorted_emotions[:3]
        ]

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "topEmotions": top_emotions,
        }


class WhisperSpeechService:
    """
    Independent modular service for Speech Transcription & Translation using OpenAI Whisper.
    Detects language (e.g., Hindi, English), transcribes source speech, and translates to English.
    """

    def __init__(self, model_name: str = "openai/whisper-tiny") -> None:
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        """Check if Whisper processor and model are loaded into memory."""
        return self.processor is not None and self.model is not None

    def load_model(self) -> None:
        """Load Whisper processor and model safely in background thread."""
        if self.is_loaded():
            return

        with self._lock:
            if self.is_loaded():
                return

            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded Whisper model '{self.model_name}' on {self.device}")
            except Exception as exc:
                logger.warning(f"Could not load Whisper model '{self.model_name}': {exc}")
                self.processor = None
                self.model = None

    def transcribe_and_translate(self, audio_16k: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio and translate to English when audio is non-English (e.g. Hindi).
        Returns:
            dict containing:
                - transcript: original text (Hindi Devanagari or English)
                - translation: English translation (or English original)
                - language: detected language code ('hi', 'en', etc.)
                - languageName: human-readable name ('Hindi', 'English', etc.)
        """
        if not self.is_loaded():
            self.load_model()

        if self.model is None or self.processor is None:
            return {
                "transcript": "Transcription model is currently initializing. Please retry in a few moments.",
                "translation": "Translation model is currently initializing.",
                "language": "unknown",
                "languageName": "Unknown",
            }

        try:
            import torch

            # Standard Whisper feature extractor supports 30s clips
            max_samples = 30 * 16000
            audio_segment = audio_16k[:max_samples]

            inputs = self.processor(audio_segment, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            with torch.no_grad():
                # 1. Generate transcription in source language with token inspection
                gen_out = self.model.generate(input_features, return_dict_in_generate=True)
                seq = gen_out.sequences[0].tolist()
                tokens = [self.processor.tokenizer.decode([t]) for t in seq]

                # Extract language code from prefix tokens: e.g. <|startoftranscript|>, <|hi|>, ...
                lang_code = "en"
                for tok in tokens[:5]:
                    cleaned = tok.replace("<|", "").replace("|>", "").strip()
                    if cleaned in LANGUAGE_NAMES or len(cleaned) == 2:
                        lang_code = cleaned
                        break

                transcript = self.processor.batch_decode(
                    gen_out.sequences,
                    skip_special_tokens=True,
                )[0].strip()

                # Clean up empty transcript fallbacks
                if not transcript:
                    transcript = "(No discernible speech detected in audio segment)"

                # 2. If non-English (e.g., Hindi), perform translation to English
                if lang_code != "en":
                    trans_out = self.model.generate(input_features, task="translate")
                    translation = self.processor.batch_decode(
                        trans_out,
                        skip_special_tokens=True,
                    )[0].strip()
                    if not translation:
                        translation = transcript
                else:
                    translation = transcript

                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())

                return {
                    "transcript": transcript,
                    "translation": translation,
                    "language": lang_code,
                    "languageName": lang_name,
                }
        except Exception as exc:
            logger.warning(f"Whisper transcription failed: {exc}")
            return {
                "transcript": "(Speech transcription unavailable for this recording)",
                "translation": "(Translation unavailable)",
                "language": "en",
                "languageName": "English",
            }


class AudioAnalysisService:
    """
    Unified coordinator service for Audio Emotion Analysis.
    Combines:
    1. Speech Emotion Recognition (SER) via SpeechEmotionRecognitionService
    2. Speech-to-Text Transcription & Translation via WhisperSpeechService
    3. Acoustic Signal Extraction (Waveform, Sample Rate, Channels, Duration)
    """

    def __init__(self) -> None:
        self.ser_service = SpeechEmotionRecognitionService()
        self.whisper_service = WhisperSpeechService()

    @property
    def model_name(self) -> str:
        """Primary SER model identifier for backward compatibility."""
        return self.ser_service.model_name

    def is_loaded(self) -> bool:
        """Check if SER model is loaded."""
        return self.ser_service.is_loaded()

    def is_whisper_loaded(self) -> bool:
        """Check if Whisper model is loaded."""
        return self.whisper_service.is_loaded()

    def load_model(self) -> None:
        """Warm up both models."""
        self.ser_service.load_model()
        self.whisper_service.load_model()

    def read_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int, int]:
        """
        Decode raw audio bytes into 1D float32 numpy array, original sample rate, and channel count.
        Supports WAV, MP3, FLAC, M4A, OGG etc.
        """
        # Try soundfile first
        try:
            import soundfile as sf

            with io.BytesIO(audio_bytes) as bio:
                data, sample_rate = sf.read(bio, dtype="float32")
                channels = data.shape[1] if data.ndim > 1 else 1
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data, int(sample_rate), channels
        except Exception as sf_err:
            logger.debug(f"soundfile decode notice: {sf_err}")

        # Try scipy.io.wavfile
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
            logger.debug(f"scipy decode notice: {scipy_err}")

        # Try wave module for standard WAV
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
                    return data, int(sample_rate), channels
        except Exception as wave_err:
            raise ValueError(f"Failed to decode audio file: {wave_err}")

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """Resample audio signal to the target sampling rate (16 kHz)."""
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
        except ImportError:
            pass

        # High-precision linear interpolation resampler
        duration = len(audio) / float(orig_sr)
        target_length = int(np.round(duration * target_sr))
        orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        target_indices = np.linspace(0, len(audio) - 1, num=target_length)
        resampled = np.interp(target_indices, orig_indices, audio)
        return resampled.astype(np.float32)

    def extract_waveform(self, audio: np.ndarray, num_bins: int = 64) -> List[float]:
        """Downsample audio into normalized peak amplitude bins for UI waveform visualizer."""
        if len(audio) == 0:
            return [0.1] * num_bins

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

        return [round(float(np.clip(v, 0.05, 1.0)), 3) for v in normalized]

    def predict(
        self,
        audio_bytes: bytes,
        filename: str = "audio_recording",
        file_size_formatted: str = "",
    ) -> Dict[str, Any]:
        """
        Execute full multimodal audio analysis pipeline:
        1. Decode acoustic stream
        2. Speech Emotion Recognition (SER) prediction
        3. Whisper Speech-to-Text Transcription & Translation
        4. Acoustic metrics & normalized waveform
        """
        raw_audio, orig_sr, channels = self.read_audio(audio_bytes)
        duration = float(len(raw_audio) / orig_sr)
        audio_16k = self.resample_audio(raw_audio, orig_sr, 16000)
        waveform = self.extract_waveform(audio_16k, num_bins=64)

        # 1. Speech Emotion Recognition (SER)
        ser_result = self.ser_service.predict_emotion(audio_16k, duration)

        # 2. Whisper Speech-to-Text & Translation
        whisper_result = self.whisper_service.transcribe_and_translate(audio_16k)

        # 3. Audio Stats
        channel_desc = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels} channels"
        ext = filename.split(".")[-1].upper() if "." in filename else "AUDIO"
        audio_stats = {
            "duration": round(duration, 2),
            "sampleRate": orig_sr,
            "channels": channels,
            "channelDesc": channel_desc,
            "format": ext,
            "fileSize": file_size_formatted,
        }

        return {
            "emotion": ser_result["emotion"],
            "confidence": ser_result["confidence"],
            "probabilities": ser_result["probabilities"],
            "topEmotions": ser_result["topEmotions"],
            "duration": round(duration, 2),
            "waveform": waveform,
            "transcript": whisper_result["transcript"],
            "translation": whisper_result["translation"],
            "language": whisper_result["language"],
            "languageName": whisper_result["languageName"],
            "audioStats": audio_stats,
            "serModel": self.ser_service.model_name,
            "transcriptionModel": self.whisper_service.model_name,
        }


# Singleton service instance exported for application usage
audio_model_service = AudioAnalysisService()
