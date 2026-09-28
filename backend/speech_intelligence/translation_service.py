import logging
from typing import Any, Dict, List
import numpy as np
from .transcript_formatter import merge_chunk_transcripts
from .whisper_service import whisper_manager

logger = logging.getLogger(__name__)


def generate_speech_translation(
    chunks: List[Dict[str, Any]],
    source_language_code: str,
    original_full_transcript: str,
) -> Dict[str, Any]:
    """
    Generate fluent English translation for multilingual speech.
    Preserves political rhetorical tone, speaker emphasis, and meaning without summarizing.
    """
    cleaned_lang = source_language_code.lower().strip()

    # If the speech is native English, return verbatim transcript
    if cleaned_lang == "en":
        logger.info("Speech is native English. Setting translation equal to original transcript.")
        return {
            "translation": original_full_transcript,
            "is_translation_needed": False,
            "notes": "Original speech is native English; no translation required.",
        }

    logger.info(
        f"Translating speech from {source_language_code} to English across {len(chunks)} chunk(s)..."
    )

    translated_chunks_text: List[str] = []

    for idx, chunk in enumerate(chunks):
        audio_segment = chunk["audio"]
        chunk_translation = whisper_manager.translate_segment(
            audio_segment,
            source_language_code=cleaned_lang,
        )
        logger.debug(
            f"Translated chunk #{idx} ({chunk['start_time']}s - {chunk['end_time']}s): {chunk_translation[:60]}..."
        )
        translated_chunks_text.append(chunk_translation)

    full_translation = merge_chunk_transcripts(translated_chunks_text)

    if not full_translation.strip():
        full_translation = "(English translation could not be generated from the audio signal)"

    logger.info(
        f"Completed translation ({len(full_translation)} characters, {len(full_translation.split())} words)"
    )

    return {
        "translation": full_translation,
        "is_translation_needed": True,
        "notes": f"Translated from {source_language_code} into fluent English.",
    }
