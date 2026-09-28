import logging
import time
from typing import Any, Dict, List
import numpy as np

from .language_detection import detect_language_from_audio, refine_language_from_text
from .preprocessing import preprocess_speech_audio
from .transcript_formatter import merge_chunk_transcripts
from .translation_service import generate_speech_translation
from .whisper_service import whisper_manager

logger = logging.getLogger("speech_intelligence")


class SpeechIntelligencePipeline:
    """
    Unified end-to-end pipeline for Political Speech Intelligence:
    1. Audio Ingestion & Preprocessing (16kHz, mono, volume normalized, chunked)
    2. Automatic Language Detection (Hindi, English, Marathi, Hinglish, etc.)
    3. Multi-Chunk Transcription in Original Spoken Language
    4. Neural English Translation preserving context and rhetorical emphasis
    """

    def __init__(self) -> None:
        self.whisper = whisper_manager

    def is_loaded(self) -> bool:
        """Check if underlying Whisper model is ready."""
        return self.whisper.is_loaded()

    def load_model(self) -> None:
        """Warm up Whisper model."""
        self.whisper.load_model()

    def extract_waveform(self, audio: np.ndarray, num_bins: int = 64) -> List[float]:
        """Generate normalized amplitude peaks for audio player visualization."""
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

    def process(
        self,
        audio_bytes: bytes,
        filename: str = "audio_recording",
        file_size_formatted: str = "",
        category: str = "Other",
    ) -> Dict[str, Any]:
        """
        Execute full speech intelligence pipeline on audio bytes.
        """
        start_time = time.time()
        logger.info(f"=== Starting Speech Intelligence pipeline for '{filename}' ===")

        # Stage 1: Audio Preprocessing
        logger.info("[Stage 1/5] Preprocessing audio stream...")
        prep_res = preprocess_speech_audio(
            audio_bytes,
            filename=filename,
            file_size_formatted=file_size_formatted,
        )
        full_audio = prep_res["full_audio"]
        chunks = prep_res["chunks"]
        stats = prep_res["stats"]

        duration = stats["duration"]
        if duration < 0.3:
            raise ValueError(f"Recording is too short ({duration:.2f}s). Please provide at least 0.5s of speech.")
        if duration > 900.0:
            raise ValueError(f"Recording is too long ({round(duration)}s). Maximum supported duration is 15 minutes.")

        waveform = self.extract_waveform(full_audio, num_bins=64)

        # Stage 2: Automatic Language Detection
        logger.info("[Stage 2/5] Detecting spoken language...")
        first_chunk_audio = chunks[0]["audio"]
        features_first = self.whisper.extract_features(first_chunk_audio)
        detected_code, detected_name, lang_conf = detect_language_from_audio(
            model=self.whisper.model,
            processor=self.whisper.processor,
            input_features=features_first,
            device=self.whisper.device,
        )

        # Stage 3: Sequential Chunk Transcription in Original Language
        logger.info(
            f"[Stage 3/5] Transcribing {len(chunks)} speech chunk(s) in original language ({detected_name})..."
        )
        raw_chunks_transcripts: List[str] = []
        chunk_details: List[Dict[str, Any]] = []

        for c in chunks:
            chunk_audio_data = c["audio"]
            text = self.whisper.transcribe_segment(
                chunk_audio_data,
                language_code=detected_code,
            )
            raw_chunks_transcripts.append(text)
            chunk_details.append(
                {
                    "index": c["index"],
                    "startTime": c["start_time"],
                    "endTime": c["end_time"],
                    "duration": c["duration"],
                    "text": text,
                }
            )
            logger.info(
                f"  Chunk #{c['index'] + 1}/{len(chunks)} [{c['start_time']}s-{c['end_time']}s]: '{text[:50]}...'"
            )

        merged_transcript = merge_chunk_transcripts(raw_chunks_transcripts)
        if not merged_transcript.strip():
            merged_transcript = "(No discernible spoken speech could be transcribed from the recording)"

        # Refine language detection using script analysis (e.g. Devanagari vs Latin)
        final_lang_code, final_lang_name = refine_language_from_text(
            merged_transcript,
            detected_code,
        )

        logger.info(f"Final speech transcript ({len(merged_transcript)} chars): {merged_transcript[:100]}...")

        # Stage 4: High-Quality English Translation
        logger.info(f"[Stage 4/5] Generating English translation from {final_lang_name}...")
        translation_res = generate_speech_translation(
            chunks=chunks,
            source_language_code=final_lang_code,
            original_full_transcript=merged_transcript,
        )
        english_translation = translation_res["translation"]
        is_translation_needed = translation_res["is_translation_needed"]

        # Stage 5: Assembly
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[Stage 5/5] Pipeline completed successfully in {elapsed}s.")

        m = int(duration // 60)
        s = int(round(duration % 60))
        formatted_duration = f"{m}:{s:02d}"

        return {
            "transcript": merged_transcript,
            "translation": english_translation,
            "isTranslationNeeded": is_translation_needed,
            "language": final_lang_code,
            "languageName": final_lang_name,
            "languageConfidence": lang_conf,
            "duration": duration,
            "formattedDuration": formatted_duration,
            "chunkCount": len(chunks),
            "chunks": chunk_details,
            "audioStats": stats,
            "waveform": waveform,
            "model": self.whisper.active_model_name,
            "processingTime": elapsed,
        }


# Singleton pipeline instance
speech_pipeline = SpeechIntelligencePipeline()
