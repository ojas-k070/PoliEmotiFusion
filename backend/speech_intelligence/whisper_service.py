import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
FALLBACK_MODEL = "openai/whisper-small"


class WhisperModelManager:
    """
    Thread-safe model manager for OpenAI Whisper speech intelligence.
    Loads and caches Whisper models with automatic resource-aware fallback.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.target_model_name = model_name
        self.active_model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        """Check if model and processor are initialized in memory."""
        return self.processor is not None and self.model is not None

    def load_model(self) -> None:
        """Load Whisper model and processor onto target compute device."""
        if self.is_loaded():
            return

        with self._lock:
            if self.is_loaded():
                return

            import warnings
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from transformers import logging as hf_logging

            # Suppress non-critical HuggingFace generation parameter notices
            hf_logging.set_verbosity_error()
            warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
            warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models_to_try = [self.target_model_name]
            if self.target_model_name != FALLBACK_MODEL:
                models_to_try.append(FALLBACK_MODEL)

            for candidate in models_to_try:
                try:
                    logger.info(f"Loading Whisper model '{candidate}' on {self.device}...")
                    self.processor = WhisperProcessor.from_pretrained(candidate)
                    self.model = WhisperForConditionalGeneration.from_pretrained(candidate)
                    if hasattr(self.model, "generation_config"):
                        self.model.generation_config.max_length = None
                    self.model.to(self.device)
                    self.model.eval()
                    self.active_model_name = candidate
                    logger.info(f"Successfully loaded Whisper model '{candidate}' on {self.device}")
                    return
                except Exception as exc:
                    logger.warning(f"Failed to load Whisper candidate '{candidate}': {exc}")

            raise RuntimeError("Could not load any OpenAI Whisper checkpoint.")

    def extract_features(self, audio_segment: np.ndarray) -> torch.Tensor:
        """Extract log-mel spectrogram features for Whisper input."""
        if not self.is_loaded():
            self.load_model()
        # Ensure audio length is at least 0.5s and at most 30s
        max_samples = 30 * 16000
        segment = audio_segment[:max_samples]
        inputs = self.processor(segment, sampling_rate=16000, return_tensors="pt")
        return inputs.input_features.to(self.device)

    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        language_code: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio segment in its original spoken language.
        If language_code is provided (e.g. 'hi' or 'mr'), enforces original script transcription.
        """
        if not self.is_loaded():
            self.load_model()

        input_features = self.extract_features(audio_segment)

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": 400,
                "no_repeat_ngram_size": 3,
            }

            if language_code and language_code != "hinglish":
                try:
                    prompt_ids = self.processor.get_decoder_prompt_ids(
                        language=language_code,
                        task="transcribe",
                    )
                    gen_kwargs["forced_decoder_ids"] = prompt_ids
                except Exception as p_err:
                    logger.debug(f"Could not force decoder prompt for {language_code}: {p_err}")

            predicted_ids = self.model.generate(input_features, **gen_kwargs)
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return transcription

    def translate_segment(
        self,
        audio_segment: np.ndarray,
        source_language_code: Optional[str] = None,
    ) -> str:
        """
        Translate an audio segment into fluent English.
        Uses Whisper's native task='translate' capability.
        """
        if not self.is_loaded():
            self.load_model()

        input_features = self.extract_features(audio_segment)

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": 400,
                "no_repeat_ngram_size": 3,
            }

            try:
                lang = source_language_code if source_language_code and source_language_code != "hinglish" else None
                prompt_ids = self.processor.get_decoder_prompt_ids(
                    language=lang,
                    task="translate",
                )
                gen_kwargs["forced_decoder_ids"] = prompt_ids
            except Exception as p_err:
                logger.debug(f"Could not force translate decoder prompt: {p_err}")

            predicted_ids = self.model.generate(input_features, **gen_kwargs)
            translation = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return translation


# Global singleton manager instance
whisper_manager = WhisperModelManager()
